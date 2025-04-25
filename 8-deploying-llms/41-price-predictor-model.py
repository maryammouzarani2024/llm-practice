

import modal 

Pricer = modal.Cls.lookup("pricer-service", "Pricer")
pricer = Pricer()
reply = pricer.price.remote("Quadcast HyperX condenser mic, connects via usb-c to your computer for crystal clear audio")
print(reply)