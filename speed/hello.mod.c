#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

BUILD_SALT;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0xfa12b18c, "module_layout" },
	{ 0xf0fda7ad, "platform_driver_unregister" },
	{ 0xbde95e6d, "__platform_driver_register" },
	{ 0x1747be18, "gpiod_set_debounce" },
	{ 0x92d5838e, "request_threaded_irq" },
	{ 0xd36c5d64, "gpiod_to_irq" },
	{ 0x6be88ee4, "devm_gpiod_get" },
	{ 0xc1514a3b, "free_irq" },
	{ 0xc5850110, "printk" },
};

MODULE_INFO(depends, "");

