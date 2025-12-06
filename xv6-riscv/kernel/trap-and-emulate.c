#include "types.h"
#include "param.h"
#include "memlayout.h"
#include "riscv.h"
#include "spinlock.h"
#include "proc.h"
#include "defs.h"
#include "stdbool.h"
#include "stdlib.h"

#define REG_COUNT 41
#define REG_SSTATUS 8
#define REG_SEPC 15
#define REG_MSTATUS 24
#define REG_MEPC 32
#define REG_STVEC 12
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

struct vm_reg {
    int     code;
    int     mode;
    uint64  val;
};

struct vm_virtual_state {
    struct vm_reg registers[REG_COUNT];
    int current_privilege;
    bool pmp_enabled;
    pagetable_t virtual_pagetable;
    struct vm_reg tmp;
};

struct vm_virtual_state vm_state;

static uint64 read_guest_reg(struct trapframe *tf, unsigned int reg);
static void write_guest_reg(struct trapframe *tf, unsigned int reg, uint64 value);
static bool is_pmp_register(int csr_index);
static void rebuild_pmp_tables(struct proc *process);
static void restrict_region(pagetable_t pagetable, uint64 start, uint64 end);
static void emit_pmp_layout(void);
static void switch_guest_pagetable(struct proc *process);
static int copy_vm_region(pagetable_t dst, pagetable_t src, uint64 start, uint64 end);
static void handle_pmp_fault(struct proc *process, uint64 scause);

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
pmp_addr_value(int idx)
{
    if(idx == 1)
        return vm_state.registers[REG_PMPADDR1].val << 2;
    else if(idx == 0)
        return vm_state.registers[REG_PMPADDR0].val << 2;
    return 0;
}

static inline uint64
pmp_cfg_byte(int idx)
{
    return (vm_state.registers[REG_PMPCFG0].val >> (idx * 8)) & 0xff;
}

int find_csr_index(unsigned int csr) {
    int i = REG_COUNT;
    while(i-- > 0) {
        if (vm_state.registers[i].code == csr)
            return i;
    }
    return -1;
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
restrict_region(pagetable_t pt, uint64 start, uint64 end)
{
    if(start >= end)
        return;
    uint64 va = PGROUNDDOWN(start);
    uint64 finish = PGROUNDUP(end);
    while(va < finish){
        pte_t *pte = walk(pt, va, 0);
        if(pte && (*pte & PTE_V))
            *pte &= ~PTE_U;
        va += PGSIZE;
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
switch_guest_pagetable(struct proc *p)
{
    if(p->vm_host_pagetable == 0)
        p->vm_host_pagetable = p->pagetable;

    if(!vm_state.pmp_enabled || vm_state.current_privilege == MACHINE_MODE){
        if(p->vm_host_pagetable)
            p->pagetable = p->vm_host_pagetable;
    } else if(p->vm_pmp_pagetable){
        p->pagetable = p->vm_pmp_pagetable;
    }
}

static void
handle_pmp_fault(struct proc *p, uint64 scause)
{
    printf("Page Fault Occured. Probably due to PMP Violation\n");
    uint64 fault_addr = r_stval();
    if(scause == INSTRUCTION_PAGE_FAULT)
        fault_addr = p->trapframe->epc;
    printf("Accessing Address: %p\n", fault_addr);
    setkilled(p);
    trap_and_emulate_init();
}

static void
rebuild_pmp_tables(struct proc *p)
{
    int has_tor = 0, i = 0;
    while(i < MAX_PMP_ENTRIES){
        if(((pmp_cfg_byte(i) >> PMP_CFG_A_SHIFT) & PMP_CFG_A_MASK) == PMP_CFG_TOR){
            has_tor = 1;
            break;
        }
        i++;
    }
    
    if(!has_tor){
        vm_state.pmp_enabled = false;
        if(p->vm_pmp_pagetable){
            proc_freepagetable(p->vm_pmp_pagetable, p->sz);
            p->vm_pmp_pagetable = 0;
        }
        switch_guest_pagetable(p);
        return;
    }

    vm_state.pmp_enabled = true;
    if(p->vm_host_pagetable == 0)
        p->vm_host_pagetable = p->pagetable;

    pagetable_t restricted = proc_pagetable(p);
    if(restricted == 0){
        setkilled(p);
        return;
    }

    if(uvmcopy(p->vm_host_pagetable, restricted, p->sz) < 0){
        proc_freepagetable(restricted, p->sz);
        setkilled(p);
        return;
    }

    if(strncmp(p->name, "vm-", 3) == 0){
        if(copy_vm_region(restricted, p->vm_host_pagetable,
                          VM_REGION_START, VM_REGION_START + VM_REGION_SIZE) < 0){
            proc_freepagetable(restricted, p->sz);
            setkilled(p);
            return;
        }
    }

    uint64 prev = 0;
    for(int i = 0; i < MAX_PMP_ENTRIES; i++){
        uint64 top = pmp_addr_value(i);
        if(top > prev && ((pmp_cfg_byte(i) >> PMP_CFG_A_SHIFT) & PMP_CFG_A_MASK) == PMP_CFG_TOR)
            restrict_region(restricted, prev, top);
        prev = top;
    }

    if(p->vm_pmp_pagetable)
        proc_freepagetable(p->vm_pmp_pagetable, p->sz);

    p->vm_pmp_pagetable = restricted;
}

int emulate_csrr(struct proc *p, unsigned int src, unsigned int dst, unsigned int csr) {
    int csr_index = find_csr_index(csr);
    if (csr_index == -1) return -1;
    if (vm_state.current_privilege < vm_state.registers[csr_index].mode)
        return -2;
    write_guest_reg(p->trapframe, dst, vm_state.registers[csr_index].val);
    p->trapframe->epc += 4;
    return 0;
}

int emulate_csrw(struct proc *p, unsigned int src, unsigned int dst, unsigned int csr) {
    int csr_index = find_csr_index(csr);
    if (csr_index == -1) return CSRW_NOT_FOUND;
    if (vm_state.current_privilege < vm_state.registers[csr_index].mode)
        return CSRW_INSUFFICIENT_PRIVILEGE;
    
    uint64 val = read_guest_reg(p->trapframe, src);
    if (csr_index == REG_MCVENDORID && val == 0x0)
        return CSRW_INVALID_MCVENDORID;
    
    vm_state.registers[csr_index].val = val;
    if (csr_index == REG_PMPADDR0 || csr_index == REG_PMPADDR1 || csr_index == REG_PMPCFG0)
        vm_state.pmp_enabled = true;
    if(is_pmp_register(csr_index))
        rebuild_pmp_tables(p);
    p->trapframe->epc += 4;
    return CSRW_SUCCESS;
}

int emulate_sret(struct proc *p) {
    if (vm_state.current_privilege < SUPERVISOR_MODE) return -1;
    unsigned long sstatus = vm_state.registers[REG_SSTATUS].val;
    int spp = (sstatus >> 8) & 0x1;
    sstatus = (sstatus & ~(1UL << 5)) | (((sstatus >> 5) & 0x1) << 1);
    sstatus &= ~(1UL << 8);
    vm_state.registers[REG_SSTATUS].val = sstatus;
    vm_state.current_privilege = spp ? SUPERVISOR_MODE : USER_MODE;
    p->trapframe->epc = vm_state.registers[REG_SEPC].val;
    switch_guest_pagetable(p);
    return 0;
}

int emulate_mret(struct proc *p) {
    if (vm_state.current_privilege < MACHINE_MODE) return -1;

    unsigned long mstatus = vm_state.registers[REG_MSTATUS].val;
    vm_state.current_privilege = ((mstatus >> 11) & 0x1) ? SUPERVISOR_MODE : USER_MODE;
    mstatus &= ~(1UL << 5);
    mstatus |= ((mstatus >> 7) & 0x1) << 3;
    vm_state.registers[REG_MSTATUS].val = mstatus;

    p->trapframe->epc = vm_state.registers[REG_MEPC].val;
    emit_pmp_layout();
    switch_guest_pagetable(p);
    return 0;
}

int emulate_ecall(struct proc *p) {
    if (vm_state.registers[REG_STVEC].val == 0) return ECALL_INVALID_STVEC;
    vm_state.registers[REG_SEPC].val = p->trapframe->epc;
    p->trapframe->epc = vm_state.registers[REG_STVEC].val;
    vm_state.current_privilege = SUPERVISOR_MODE;
    switch_guest_pagetable(p);
    return ECALL_SUCCESS;
}

void trap_and_emulate(void) {
    struct proc *p = myproc();
    uint64 scause = r_scause();

    if(scause == LOAD_PAGE_FAULT || scause == STORE_PAGE_FAULT || scause == INSTRUCTION_PAGE_FAULT){
        handle_pmp_fault(p, scause);
        return;
    }

    uint64 va = r_sepc();
    uint64 pa = walkaddr(p->pagetable, va) | (va & 0xFFF);

    if (pa == 0) {
        printf("Invalid virtual address: %p\n", va);
        setkilled(p);
        return;
    }

    uint32 inst = *((uint32 *)(pa));
    uint32 op = inst & 0x7F, dst = (inst >> 7) & 0x1F, funct3 = (inst >> 12) & 0x7;
    uint32 src = (inst >> 15) & 0x1F, csr = (inst >> 20) & 0xFFF;

    if (funct3 == 0x0 && csr == 0x0) {
        printf("(EC at %p)\n", p->trapframe->epc);
        if (emulate_ecall(p) != 0) {
            setkilled(p);
        }
        return;
    }

    printf("(PI at %p) op = %x, rd = %x, funct3 = %x, rs1 = %x, uimm = %x\n",
           va, op, dst, funct3, src, csr);

    switch (funct3) {
        case 0x0:
            switch (csr) {
                case 0x102:
                    if (emulate_sret(p) != 0) {
                        setkilled(p);
                    }
                    return;
                case 0x302:
                    if (emulate_mret(p) != 0) {
                        setkilled(p);
                    }
                    return;
                default:
                    break;
            }
            break;
        case 0x1:
            if (emulate_csrw(p, src, dst, csr) == 0) {
                return;
            }
            break;
        case 0x2:
            if (emulate_csrr(p, src, dst, csr) == 0) {
                return;
            }
            break;
    }

    setkilled(p);
    trap_and_emulate_init();
}

void trap_and_emulate_init(void) {
    vm_state.pmp_enabled = false;

    vm_state.registers[0] = (struct vm_reg){0x000, 0, 0};
    vm_state.registers[1] = (struct vm_reg){0x004, 0, 0};
    vm_state.registers[3] = (struct vm_reg){0x040, 0, 0};
    vm_state.registers[4] = (struct vm_reg){0x041, 0, 0};
    vm_state.registers[5] = (struct vm_reg){0x042, 0, 0};
    vm_state.registers[6] = (struct vm_reg){0x043, 0, 0};
    vm_state.registers[7] = (struct vm_reg){0x044, 0, 0};
    vm_state.registers[8] = (struct vm_reg){0x100, 1, 0};
    vm_state.registers[9] = (struct vm_reg){0x102, 1, 0};
    vm_state.registers[10] = (struct vm_reg){0x103, 1, 0};
    vm_state.registers[11] = (struct vm_reg){0x104, 1, 0};
    vm_state.registers[12] = (struct vm_reg){0x105, 1, 0};
    vm_state.registers[13] = (struct vm_reg){0x106, 1, 0};
    vm_state.registers[14] = (struct vm_reg){0x140, 1, 0};
    vm_state.registers[15] = (struct vm_reg){0x141, 1, 0};
    vm_state.registers[16] = (struct vm_reg){0x142, 1, 0};
    vm_state.registers[17] = (struct vm_reg){0x143, 1, 0};
    vm_state.registers[18] = (struct vm_reg){0x144, 1, 0};
    vm_state.registers[19] = (struct vm_reg){0x180, 1, 0};
    vm_state.registers[20] = (struct vm_reg){0xf11, 1, 0x637365353336};
    vm_state.registers[21] = (struct vm_reg){0xf12, 2, 0};
    vm_state.registers[22] = (struct vm_reg){0xf13, 2, 0};
    vm_state.registers[23] = (struct vm_reg){0xf14, 2, 0};
    vm_state.registers[24] = (struct vm_reg){0x300, 2, 0};
    vm_state.registers[25] = (struct vm_reg){0x301, 2, 0};
    vm_state.registers[26] = (struct vm_reg){0x302, 2, 0};
    vm_state.registers[27] = (struct vm_reg){0x303, 2, 0};
    vm_state.registers[28] = (struct vm_reg){0x304, 2, 0};
    vm_state.registers[29] = (struct vm_reg){0x305, 2, 0};
    vm_state.registers[30] = (struct vm_reg){0x306, 2, 0};
    vm_state.registers[31] = (struct vm_reg){0x340, 2, 0};
    vm_state.registers[32] = (struct vm_reg){0x341, 2, 0};
    vm_state.registers[33] = (struct vm_reg){0x342, 2, 0};
    vm_state.registers[34] = (struct vm_reg){0x343, 2, 0};
    vm_state.registers[35] = (struct vm_reg){0x344, 2, 0};
    vm_state.registers[36] = (struct vm_reg){0x34a, 2, 0};
    vm_state.registers[37] = (struct vm_reg){0x34b, 2, 0};
    vm_state.registers[38] = (struct vm_reg){0x3a0, 2, 0};
    vm_state.registers[39] = (struct vm_reg){0x3b0, 2, 0};
    vm_state.registers[40] = (struct vm_reg){0x3b1, 2, 0};

    vm_state.current_privilege = MACHINE_MODE;
    vm_state.virtual_pagetable = NULL;
}

