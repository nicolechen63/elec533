/*
 * hello.c
 * 
 * Beaglebone AI-64
 * 
 * Driver for the speed encoder
 * Input - pin P8_03 from speed encoder
 *
 *  The input from the speed encoder will trigger interrupt service routine
 *  It will save 5 recent time intervals and the interrupt count as parameters 
 *   for the system control logic
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
static unsigned long count; // count interrupt
static unsigned long t, t0, t1; // time invertal
static unsigned long x0 = 0, x1 = 0, x2 = 0, x3 = 0, x4 = 0; // to hold 5 time intervals

// parameters output for control script
module_param(count, ulong, S_IRUGO|S_IWUSR);
module_param(x0, ulong, S_IRUGO|S_IWUSR);
module_param(x1, ulong, S_IRUGO|S_IWUSR);
module_param(x2, ulong, S_IRUGO|S_IWUSR);
module_param(x3, ulong, S_IRUGO|S_IWUSR);
module_param(x4, ulong, S_IRUGO|S_IWUSR);

// interrupt service routine
//  - save recent 5 time intervals
static irq_handler_t gpio_irq_handler(unsigned int irq, void *dev_id, struct pt_regs *regs) {
	count++; // increase the count

	t1 = jiffies; // get current time
	t = t1 - t0; // time interval

	// save the current 5 values of time interval, newest value appears at x0
	x4 = x3;
	x3 = x2;
	x2 = x1;
	x1 = x0;
	x0 = t;

	t0 = t1; // set previous time for next

	printk(KERN_INFO "count = %ld, t = %ld\n", count, t);

	return (irq_handler_t) IRQ_HANDLED; 
}

// probe function
static int led_probe(struct platform_device *pdev)
{
	printk(KERN_INFO "led_probe\n");

	// intialize and get input gpio pins
	gpiod_but = devm_gpiod_get(&pdev->dev, "userbutton", GPIOD_IN);

	irq_number = gpiod_to_irq(gpiod_but); // get the irq number from the gpio input
	
	// request interrupt service routine (@ rising edge)
	if (0 != request_irq(irq_number, (irq_handler_t) gpio_irq_handler, IRQF_TRIGGER_RISING, "gpiod_driver", NULL)){ 
		printk(KERN_INFO "fail to request irq, with irq_number = %d\n", irq_number);
		return -1; // exit if unable to request for interrupt
	} else {
		printk(KERN_INFO "request irq success, with irq_number = %d\n", irq_number);
	}

	gpiod_set_debounce(gpiod_but, 1000000); // debounce

	// initialize counter for speed encoder
	count = 0;
	t = 0;
	t0 = jiffies;
	t1 = 0;

	return 0;
}

// remove function
static int led_remove(struct platform_device *pdev)
{	//free the irq and print a message
	printk(KERN_INFO "free gpio_irq\n");
	free_irq(irq_number, NULL); // free irq
	return 0;
}

static struct of_device_id matchy_match[] = {
    {	.compatible = "hello"},
    {/* leave alone - keep this here (end node) */},
};

// platform driver object
static struct platform_driver autocar_driver = {
	.probe	 = led_probe,
	.remove	 = led_remove,
	.driver	 = {
		.name  = "The Rock: this name doesn't even matter",
		.owner = THIS_MODULE,
		.of_match_table = matchy_match,
	},
};

module_platform_driver(autocar_driver);

MODULE_DESCRIPTION("553 final project");
MODULE_AUTHOR("Let's & Go");
MODULE_LICENSE("GPL v2");
MODULE_ALIAS("platform:autocar_driver");
