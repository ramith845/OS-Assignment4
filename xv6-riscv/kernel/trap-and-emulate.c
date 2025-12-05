#include "types.h"
#include "param.h"
#include "memlayout.h"
#include "riscv.h"
#include "spinlock.h"
#include "proc.h"
#include "defs.h"
#include "stdbool.h"
#include "stdlib.h"

//defining registers and modes
#define REG_COUNT 41
#define REG_SSTATUS 8
#define REG_SEPC 15
#define REG_MSTATUS 24
#define REG_MEPC 32
#define REG_STVEC 12
#define REG_SATP 19
#define REG_MCVENDORID 20
#define REG_PMPCFG0 38
#define REG_PMPADDR0 39
#define REG_PMPADDR1 40
#define MACHINE_MODE 2
#define SUPERVISOR_MODE 1
#define USER_MODE 0
#define CSRW_SUCCESS 0
#define CSRW_NOT_FOUND -1
#define CSRW_INVALID_MCVENDORID -2
#define CSRW_INSUFFICIENT_PRIVILEGE -3
#define ECALL_SUCCESS 0
#define ECALL_INVALID_STVEC -1
#define PMP_CFG_A_SHIFT 3
#define PMP_CFG_A_MASK 0x3
#define PMP_CFG_TOR 0x1
#define MAX_PMP_ENTRIES 2
#define VM_REGION_START 0x80000000ULL
#define VM_REGION_PAGES 1024
#define VM_REGION_SIZE ((uint64)VM_REGION_PAGES * PGSIZE)
#define INSTRUCTION_PAGE_FAULT 12
#define LOAD_PAGE_FAULT 13
#define STORE_PAGE_FAULT 15

// Struct to keep VM registers (Sample; feel free to change.)
typedef struct {
    int reg_code;
    int privilege_level;
    uint64 value;
    uint8 flags;
    uint64 last_update;
} VMRegister;

typedef struct {
    VMRegister registers[REG_COUNT];
    int current_privilege;
    bool pmp_enabled;
    pagetable_t virtual_pagetable;
} VMState;

VMState vm_state;

static uint64 read_guest_reg(struct trapframe *tf, unsigned int reg);
static void write_guest_reg(struct trapframe *tf, unsigned int reg, uint64 value);
static bool is_pmp_register(int csr_index);
static void rebuild_pmp_tables(struct proc *process);
static void restrict_region(pagetable_t pagetable, uint64 start, uint64 end);
static void emit_pmp_layout(void);
static void switch_guest_pagetable(struct proc *process);
static int copy_vm_region(pagetable_t dst, pagetable_t src, uint64 start, uint64 end);
static void handle_pmp_fault(struct proc *process, uint64 scause);


void initialize_register(int index, uint32 code, int mode, uint64 value) {
    vm_state.registers[index] = (VMRegister){.reg_code = code, .privilege_level = mode, .value = value, .flags = 0, .last_update = 0};
}

static uint64*
guest_reg_ptr(struct trapframe *tf, unsigned int reg)
{
    if (reg == 0)
        return 0;
    return &(tf->ra) + (reg - 1);
}

static uint64
read_guest_reg(struct trapframe *tf, unsigned int reg)
{
    uint64 *ptr = guest_reg_ptr(tf, reg);
    return ptr ? *ptr : 0;
}

static void
write_guest_reg(struct trapframe *tf, unsigned int reg, uint64 value)
{
    uint64 *ptr = guest_reg_ptr(tf, reg);
    if (ptr)
        *ptr = value;
}

static inline uint64
pmp_cfg_byte(int index)
{
    return (vm_state.registers[REG_PMPCFG0].value >> (index * 8)) & 0xff;
}

static inline uint64
pmp_addr_value(int index)
{
    if(index == 0)
        return vm_state.registers[REG_PMPADDR0].value << 2;
    if(index == 1)
        return vm_state.registers[REG_PMPADDR1].value << 2;
    return 0;
}

bool is_invalid_mcvendorid_write(int csr_index, uint64 value) { //check for invalid write
    return (csr_index == REG_MCVENDORID) && (value == 0x0);
}

void enable_pmp_if_needed(int csr_index) { //check for pmp
    if (csr_index == REG_PMPADDR0 || csr_index == REG_PMPADDR1 || csr_index == REG_PMPCFG0) {
        vm_state.pmp_enabled = true;
    }
}

int locate_csr(unsigned int csr_code) {
    for (int i = 0; i < REG_COUNT; i++) {
        if (vm_state.registers[i].reg_code == csr_code) {
            return i;
        }
    }
    return -1;  
}

void print_instruction(uint64 virtual_address, uint32 opcode, uint32 rd, uint32 funct3, uint32 rs1, uint32 uimm, const char* prefix) {
    printf("(%s at %p) op = %x, rd = %x, funct3 = %x, rs1 = %x, uimm = %x\n",
           prefix, virtual_address, opcode, rd, funct3, rs1, uimm);
}

bool is_pmp_register(int csr_index) {
    return csr_index == REG_PMPCFG0 || csr_index == REG_PMPADDR0 || csr_index == REG_PMPADDR1;
}

static int
copy_vm_region(pagetable_t dst, pagetable_t src, uint64 start, uint64 end)
{
    for(uint64 va = start; va < end; va += PGSIZE){
        pte_t *pte = walk(src, va, 0);
        if(pte == 0 || (*pte & PTE_V) == 0)
            continue;
        char *mem = kalloc();
        if(mem == 0)
            return -1;
        memmove(mem, (char*)PTE2PA(*pte), PGSIZE);
        if(mappages(dst, va, PGSIZE, (uint64)mem, PTE_FLAGS(*pte)) != 0){
            kfree(mem);
            return -1;
        }
    }
    return 0;
}

static void
restrict_region(pagetable_t pagetable, uint64 start, uint64 end)
{
    if(end <= start)
        return;
    uint64 begin = PGROUNDDOWN(start);
    uint64 finish = PGROUNDUP(end);
    for(uint64 va = begin; va < finish; va += PGSIZE){
        pte_t *pte = walk(pagetable, va, 0);
        if(pte && (*pte & PTE_V)){
            *pte &= ~PTE_U;
        }
    }
}

static void
emit_pmp_layout(void)
{
    if(!vm_state.pmp_enabled)
        return;
    uint64 prev = 0;
    for(int i = 0; i < MAX_PMP_ENTRIES; i++){
        uint64 top = pmp_addr_value(i);
        if(top > prev){
            uint64 perm = pmp_cfg_byte(i);
            printf("Region: %p to %p, Perm: %p\n", prev, top, perm);
        }
        prev = top;
    }
}

static void
switch_guest_pagetable(struct proc *process)
{
    if(process->vm_host_pagetable == 0)
        process->vm_host_pagetable = process->pagetable;

    if(!vm_state.pmp_enabled || vm_state.current_privilege == MACHINE_MODE){
        if(process->vm_host_pagetable)
            process->pagetable = process->vm_host_pagetable;
    } else if(process->vm_pmp_pagetable){
        process->pagetable = process->vm_pmp_pagetable;
    }
}

static void
handle_pmp_fault(struct proc *process, uint64 scause)
{
    printf("Page Fault Occured. Probably due to PMP Violation\n");
    uint64 fault_addr = r_stval();
    if(scause == INSTRUCTION_PAGE_FAULT)
        fault_addr = process->trapframe->epc;
    printf("Accessing Address: %p\n", fault_addr);
    setkilled(process);
    trap_and_emulate_init();
}

static bool
has_active_tor_entry(void)
{
    for(int i = 0; i < MAX_PMP_ENTRIES; i++){
        uint64 cfg = pmp_cfg_byte(i);
        uint64 mode = (cfg >> PMP_CFG_A_SHIFT) & PMP_CFG_A_MASK;
        if(mode == PMP_CFG_TOR)
            return true;
    }
    return false;
}

static void
rebuild_pmp_tables(struct proc *process)
{
    if(!has_active_tor_entry()){
        vm_state.pmp_enabled = false;
        if(process->vm_pmp_pagetable){
            proc_freepagetable(process->vm_pmp_pagetable, process->sz);
            process->vm_pmp_pagetable = 0;
        }
        switch_guest_pagetable(process);
        return;
    }

    vm_state.pmp_enabled = true;
    if(process->vm_host_pagetable == 0)
        process->vm_host_pagetable = process->pagetable;

    pagetable_t restricted = proc_pagetable(process);
    if(restricted == 0){
        setkilled(process);
        return;
    }

    if(uvmcopy(process->vm_host_pagetable, restricted, process->sz) < 0){
        proc_freepagetable(restricted, process->sz);
        setkilled(process);
        return;
    }

    if(strncmp(process->name, "vm-", 3) == 0){
        if(copy_vm_region(restricted, process->vm_host_pagetable,
                          VM_REGION_START, VM_REGION_START + VM_REGION_SIZE) < 0){
            proc_freepagetable(restricted, process->sz);
            setkilled(process);
            return;
        }
    }

    uint64 prev = 0;
    for(int i = 0; i < MAX_PMP_ENTRIES; i++){
        uint64 top = pmp_addr_value(i);
        if(top > prev){
            uint64 cfg = pmp_cfg_byte(i);
            uint64 mode = (cfg >> PMP_CFG_A_SHIFT) & PMP_CFG_A_MASK;
            if(mode == PMP_CFG_TOR)
                restrict_region(restricted, prev, top);
        }
        prev = top;
    }

    if(process->vm_pmp_pagetable)
        proc_freepagetable(process->vm_pmp_pagetable, process->sz);

    process->vm_pmp_pagetable = restricted;
}

//csrr handler
int handle_csrr(struct proc *process, unsigned int source_reg, unsigned int dest_reg, unsigned int csr_code) {
    int csr_index = locate_csr(csr_code);
    if (csr_index == -1) return -1;

    if (vm_state.current_privilege >= vm_state.registers[csr_index].privilege_level) {
        uint64 csr_value = vm_state.registers[csr_index].value;
        write_guest_reg(process->trapframe, dest_reg, csr_value);
    } else {
        return -2; 
    }
    process->trapframe->epc += 4;
    return 0;
}

//csrw handler
int handle_csrw(struct proc *process, unsigned int source_reg, unsigned int dest_reg, unsigned int csr_code) {
    int csr_index = locate_csr(csr_code);
    if (csr_index == -1) return CSRW_NOT_FOUND;

    if (vm_state.current_privilege >= vm_state.registers[csr_index].privilege_level) {
        uint64 value = read_guest_reg(process->trapframe, source_reg);

        if (is_invalid_mcvendorid_write(csr_index, value)) {
            return CSRW_INVALID_MCVENDORID;
        }

        enable_pmp_if_needed(csr_index); //pmp enabled
        vm_state.registers[csr_index].value = value;
        if(is_pmp_register(csr_index))
            rebuild_pmp_tables(process);
    } else {
        return CSRW_INSUFFICIENT_PRIVILEGE;
    }
    process->trapframe->epc += 4;
    return CSRW_SUCCESS;
}

//sret handler
int handle_sret(struct proc *process) {
    if (vm_state.current_privilege < SUPERVISOR_MODE) return -1;

    unsigned long status_register = vm_state.registers[REG_SSTATUS].value;
    unsigned long spp_bit = (status_register >> 8) & 0x1;

    status_register &= ~(1UL << 5);  //clearing spie
    status_register |= ((status_register >> 5) & 0x1) << 1;  // Set SIE
    status_register &= ~(1UL << 8);  //clearing SPP 

    vm_state.current_privilege = spp_bit ? SUPERVISOR_MODE : USER_MODE;
    vm_state.registers[REG_SSTATUS].value = status_register;

    process->trapframe->epc = vm_state.registers[REG_SEPC].value;
    switch_guest_pagetable(process);
    return 0;
}

//handling mret
int handle_mret(struct proc *process) {
    if (vm_state.current_privilege < MACHINE_MODE) return -1;

    unsigned long status_register = vm_state.registers[REG_MSTATUS].value;
    unsigned long previous_mode = (status_register >> 11) & 0x1;

    status_register &= ~(1UL << 5);  //clearing mpp bits
    status_register |= ((status_register >> 7) & 0x1) << 3;  // Set MIE
    vm_state.current_privilege = previous_mode ? SUPERVISOR_MODE : USER_MODE;
    vm_state.registers[REG_MSTATUS].value = status_register;

    process->trapframe->epc = vm_state.registers[REG_MEPC].value;
    emit_pmp_layout();
    switch_guest_pagetable(process);
    return 0;
}

//ecall handler
int handle_ecall(struct proc *process) {
    if (vm_state.registers[REG_STVEC].value == 0) return ECALL_INVALID_STVEC;

    vm_state.registers[REG_SEPC].value = process->trapframe->epc;
    process->trapframe->epc = vm_state.registers[REG_STVEC].value;
    vm_state.current_privilege = SUPERVISOR_MODE;
    switch_guest_pagetable(process);
    return ECALL_SUCCESS;
}

void trap_and_emulate(void) {
/* Comes here when a VM tries to execute a supervisor instruction. */
    struct proc *process = myproc();
    uint64 scause = r_scause();

    if(scause == LOAD_PAGE_FAULT || scause == STORE_PAGE_FAULT){
        handle_pmp_fault(process, scause);
        return;
    }
    if(scause == INSTRUCTION_PAGE_FAULT){
        handle_pmp_fault(process, scause);
        return;
    }
    
/* Retrieve all required values from the instruction */
    uint64 virtual_address = r_sepc();
    uint64 physical_address = walkaddr(process->pagetable, virtual_address) | (virtual_address & 0xFFF);

    if (physical_address == 0) {
        printf("Invalid virtual address: %p\n", virtual_address);
        setkilled(process);
        return;
    }

    uint32 instruction = *((uint32 *)(physical_address));
    uint32 opcode = instruction & 0x7F;
    uint32 rd = (instruction >> 7) & 0x1F;
    uint32 funct3 = (instruction >> 12) & 0x7;
    uint32 rs1 = (instruction >> 15) & 0x1F;
    uint32 uimm = (instruction >> 20) & 0xFFF;

    // In your ECALL, add the following for prints
// struct proc* p = myproc();
// printf("(EC at %p)\n", p->trapframe->epc);


    if (funct3 == 0x0 && uimm == 0x0) {
        printf("(EC at %p)\n", process->trapframe->epc);
    }

    /* Print the statement */
    printf("(PI at %p) op = %x, rd = %x, funct3 = %x, rs1 = %x, uimm = %x\n",
           virtual_address, opcode, rd, funct3, rs1, uimm);

    if (funct3 == 0x0 && uimm == 0x0) {
        if (handle_ecall(process) != 0) {
            setkilled(process);
        }
        return;
    }

    
    switch (funct3) {
        case 0x0:
            //handling SRET and MRET
            switch (uimm) {
                case 0x102:
                    if (handle_sret(process) != 0) {
                        setkilled(process);
                    }
                    return;
                case 0x302:
                    if (handle_mret(process) != 0) {
                        setkilled(process);
                    }
                    return;
                default:
                    break;
            }
            break;
         //handling csrw
        case 0x1:
            if (handle_csrw(process, rs1, rd, uimm) == 0) {
                return;
            }
            break;
	//handling csrr
        case 0x2:
            if (handle_csrr(process, rs1, rd, uimm) == 0) {
                return;
            }
            break;
    }

    //handling other instructions
    //printf("Unhandled or invalid instruction at virtual_address: %p\n", virtual_address);
    setkilled(process);
    trap_and_emulate_init();
}


// Initialization
void trap_and_emulate_init(void) {
/* Create and initialize all state for the VM */
    vm_state.pmp_enabled = false;

    initialize_register(0, 0x000, 0, 0);
    initialize_register(1, 0x004, 0, 0);
    initialize_register(3, 0x040, 0, 0);
    initialize_register(4, 0x041, 0, 0);
    initialize_register(5, 0x042, 0, 0);
    initialize_register(6, 0x043, 0, 0);
    initialize_register(7, 0x044, 0, 0);
    initialize_register(8, 0x100, 1, 0);
    initialize_register(9, 0x102, 1, 0);
    initialize_register(10, 0x103, 1, 0);
    initialize_register(11, 0x104, 1, 0);
    initialize_register(12, 0x105, 1, 0);
    initialize_register(13, 0x106, 1, 0);
    initialize_register(14, 0x140, 1, 0);
    initialize_register(15, 0x141, 1, 0);
    initialize_register(16, 0x142, 1, 0);
    initialize_register(17, 0x143, 1, 0);
    initialize_register(18, 0x144, 1, 0);
    initialize_register(19, 0x180, 1, 0);
    initialize_register(20, 0xf11, 1, 0x637365353336); //CSE536 hexadec code
    initialize_register(21, 0xf12, 2, 0);
    initialize_register(22, 0xf13, 2, 0);
    initialize_register(23, 0xf14, 2, 0);
    initialize_register(24, 0x300, 2, 0);
    initialize_register(25, 0x301, 2, 0);
    initialize_register(26, 0x302, 2, 0);
    initialize_register(27, 0x303, 2, 0);
    initialize_register(28, 0x304, 2, 0);
    initialize_register(29, 0x305, 2, 0);
    initialize_register(30, 0x306, 2, 0);
    initialize_register(31, 0x340, 2, 0);
    initialize_register(32, 0x341, 2, 0);
    initialize_register(33, 0x342, 2, 0);
    initialize_register(34, 0x343, 2, 0);
    initialize_register(35, 0x344, 2, 0);
    initialize_register(36, 0x34a, 2, 0);
    initialize_register(37, 0x34b, 2, 0);
    initialize_register(38, 0x3a0, 2, 0);
    initialize_register(39, 0x3b0, 2, 0);
    initialize_register(40, 0x3b1, 2, 0);

    vm_state.current_privilege = MACHINE_MODE;
    vm_state.virtual_pagetable = NULL;
}

