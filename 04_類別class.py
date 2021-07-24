
# class

class Man:

    def __init__(self, name):
        self.name = name
        print("Initilized!")

    def hello(self):
        print("Hello " + self.name + "!")
    
    def thank(self):
        print(self.name + "says /'thank you!/'")

    def goodbye(self):
        print("Good-bye " + self.name + "!")
        

m = Man("Dell")
m.hello()
m.goodbye()
