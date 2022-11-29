/*
 * hello.c
 * 
 * Beaglebone AI-64
 * 
 * This is the driver for the speed encoder
 * Input
 *  - P8_03
 *
 *  The input button pushed will toggle the led by interrupt service routine
 *
 */
#include <linux/module.h>
#include <linux/of_device.h>
#include <linux/kernel.h>
#include <linux/gpio/consumer.h>
#include <linux/platform_device.h>

#include <linux/interrupt.h>
#include <linux/sched.h>

// variables
static int irq_number;
static struct gpio_desc *gpiod_but;
static unsigned long count;
static unsigned long t, t0, t1;

// parameter
module_param(count, ulong, S_IRUGO|S_IWUSR);
module_param(t, ulong, S_IRUGO|S_IWUSR);

/* ADD THE INTERRUPT SERVICE ROUTINE HERE */
// https://github.com/Johannes4Linux/Linux_Driver_Tutorial/blob/main/11_gpio_irq/gpio_irq.c
static irq_handler_t gpio_irq_handler(unsigned int irq, void *dev_id, struct pt_regs *regs) {
	//printk(KERN_INFO "button pressed"); // print the led state

	count++;
	t1 = jiffies;
	t = t1 - t0;
	printk(KERN_INFO "count = %ld, period = %ld",count, t);
	t0 = t1;

	return (irq_handler_t) IRQ_HANDLED; 
}

// probe function
static int led_probe(struct platform_device *pdev)
{
	printk(KERN_INFO "led_probe\n");

	// intialize and get input gpio pins
	gpiod_but = devm_gpiod_get(&pdev->dev, "userbutton", GPIOD_IN);

	irq_number = gpiod_to_irq(gpiod_but); // get the irq number from the gpio input
	
	// request interrupt service routine
	//  - toggled on rising edge
	//	- exit if unable to request for interrupt
	if (0 != request_irq(irq_number, (irq_handler_t) gpio_irq_handler, IRQF_TRIGGER_RISING, "gpiod_driver", NULL)){ 
		printk(KERN_INFO "fail to request irq, with irq_number = %d\n", irq_number);
		return -1;
	} else {
		printk(KERN_INFO "request irq success, with irq_number = %d\n", irq_number);
	}

	gpiod_set_debounce(gpiod_but, 1000000); // debounce

	// initialize counter for speed encoding
	count = 0;
	t = HZ;
	t0 = jiffies;

	printk("one second: %ld\n", t);
	printk("jiffies: %ld\n", t0);

	return 0;
}

// remove function
static int led_remove(struct platform_device *pdev)
{
	/* INSERT: Free the irq and print a message */
	printk(KERN_INFO "free gpio_irq\n");
	free_irq(irq_number, NULL); // free irq
	return 0;
}

static struct of_device_id matchy_match[] = {
    {	.compatible = "hello"},
    {/* leave alone - keep this here (end node) */},
};

// platform driver object
static struct platform_driver adam_driver = {
	.probe	 = led_probe,
	.remove	 = led_remove,
	.driver	 = {
		.name  = "The Rock: this name doesn't even matter",
		.owner = THIS_MODULE,
		.of_match_table = matchy_match,
	},
};

module_platform_driver(adam_driver);

MODULE_DESCRIPTION("553\'s finest");
MODULE_AUTHOR("Let's & Go");
MODULE_LICENSE("GPL v2");
MODULE_ALIAS("platform:adam_driver");
