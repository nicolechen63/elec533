cmd_/home/debian/elec533/speed/modules.order := {   echo /home/debian/elec533/speed/hello.ko; :; } | awk '!x[$$0]++' - > /home/debian/elec533/speed/modules.order
