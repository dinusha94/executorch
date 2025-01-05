__STACK_SIZE = 0x00008000;
__HEAP_SIZE = 0x00008000;
MEMORY
{
  ITCM (rx) : ORIGIN = 0x10000000, LENGTH = 0x00100000
  BRAM (rw) : ORIGIN = 0x11000000, LENGTH = 0x00100000
  DTCM (rw) : ORIGIN = 0x30000000, LENGTH = 0x00080000
  SRAM (rw) : ORIGIN = 0x31000000, LENGTH = 0x00200000
  QSPI (rw) : ORIGIN = 0x38000000, LENGTH = 0x00800000
  DDR (rwx) : ORIGIN = 0x70000000, LENGTH = 0x60000000
}
PHDRS
{
    rom_exec PT_LOAD;
    rom_dram PT_LOAD;
    data PT_LOAD;
    null PT_NULL;
}
ENTRY(Reset_Handler)
SECTIONS
{
  .text :
  {
    KEEP(*(.vectors))
    *(.text*)
    KEEP(*(.init))
    KEEP(*(.fini))
    *crtbegin.o(.ctors)
    *crtbegin?.o(.ctors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .ctors)
    *(SORT(.ctors.*))
    *(.ctors)
    *crtbegin.o(.dtors)
    *crtbegin?.o(.dtors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .dtors)
    *(SORT(.dtors.*))
    *(.dtors)
    *(.rodata*)
    KEEP(*(.eh_frame*))
  } > ITCM :rom_exec
  .ARM.extab :
  {
    *(.ARM.extab* .gnu.linkonce.armextab.*)
  } > ITCM :rom_exec
  .ARM.exidx :
  {
  __exidx_start = .;
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
  __exidx_end = .;
  } > ITCM :rom_exec
  .copy.table :
  {
    . = ALIGN(4);
    __copy_table_start__ = .;
    LONG (__etext)
    LONG (__data_start__)
    LONG ((__data_end__ - __data_start__) / 4)
    LONG (__eddr_data)
    LONG (__sram_data_start__)
    LONG ((__sram_data_end__ - __sram_data_start__) / 4)
    __copy_table_end__ = .;
  } > ITCM :rom_exec
  .zero.table :
  {
    . = ALIGN(4);
    __zero_table_start__ = .;
    LONG (__bss_start__)
    LONG ((__bss_end__ - __bss_start__) / 4)
    __zero_table_end__ = .;
  __etext = ALIGN (4);
  } > ITCM :rom_exec
  .data : AT(__etext)
  {
    __data_start__ = .;
    *(vtable)
    *(.data)
    *(.data.*)
    . = ALIGN(4);
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP(*(.preinit_array))
    PROVIDE_HIDDEN (__preinit_array_end = .);
    . = ALIGN(4);
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP(*(SORT(.init_array.*)))
    KEEP(*(.init_array))
    PROVIDE_HIDDEN (__init_array_end = .);
    . = ALIGN(4);
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP(*(SORT(.fini_array.*)))
    KEEP(*(.fini_array))
    PROVIDE_HIDDEN (__fini_array_end = .);
    KEEP(*(.jcr*))
    . = ALIGN(4);
    __data_end__ = .;
  } > DTCM :data
  .sram.bss :
  {
    . = ALIGN(16);
    *(.bss.ethosu_scratch);
    *.(output_data_sec)
  } > SRAM :null
  .ddr :
  {
    . = ALIGN(16);
    *(.bss.tensor_arena)
    . = ALIGN(4);
    *(input_data_sec)
    . = ALIGN(16);
    *(network_model_sec)
    * (expected_output_data_sec)
    * (sec_command_stream, sec_weight_data, sec_input_data)
    * (ethosu_core_in_queue)
    * (ethosu_core_out_queue)
    . = ALIGN(4);
  } > DDR :rom_dram
  __eddr_data = ALIGN (4) ;
  .sram.data : {
    __sram_data_start__ = .;
    *(.sram.data)
    __sram_data_end__ = .;
  } > BRAM AT >DDR :rom_dram
  .bss :
  {
    . = ALIGN(4);
    __bss_start__ = .;
    *(.bss)
    *(.bss.*)
    *(COMMON)
    . = ALIGN(4);
    __bss_end__ = .;
  } > DTCM :null
  .heap (COPY) :
  {
    . = ALIGN(8);
    __end__ = .;
    PROVIDE(end = .);
    . = . + __HEAP_SIZE;
    . = ALIGN(8);
    __HeapLimit = .;
  } > DTCM :null
  .stack (ORIGIN(DTCM) + LENGTH(DTCM) - __STACK_SIZE) (COPY) :
  {
    . = ALIGN(8);
    __StackLimit = .;
    . = . + __STACK_SIZE;
    . = ALIGN(8);
    __StackTop = .;
  } > DTCM :null
  PROVIDE(__stack = __StackTop);
  ASSERT(__StackLimit >= __HeapLimit, "region DTCM overflowed with stack")
}
