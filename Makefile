# Modified from:
# Dr. Derek Molloy, School of Electronic Engineering, Dublin City University,
# Ireland. URL: http://derekmolloy.ie/writing-a-linux-kernel-module-part-1-introduction/
PWD=$(shell pwd)
KERNEL_BUILD=/lib/modules/$(shell uname -r)/build

obj-m+=hello.o

all:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) modules
	sudo cp hello.ko /lib/modules/$(shell uname -r)/kernel/drivers/misc/
# copy the module file into the modules
clean:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) clean
