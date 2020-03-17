import random #libraries
import time


print("Hello World!")

print("*****************************************")
print("****************user input********************")
userName = input("What is your name?") #Python 3.X or later
print("hello " + userName) 

counter = 3          # An integer assignment
weight   = 80.5       # A floating point
name    = "David"       # A string
temperatures = [18, 15, 20, 22, 17] # A list

print("*****************************************")
print("****************dctionariesi********************")
grades = {'peter': 79, 'john': 84, 'scott': 92} # A dictionary of key value pairs
print("peters grade is" + str(grades['peter']))     # Prints value for 'peter' key - wrap in up in a str

print(grades)

print("*****************************************")
print("************random numbers***************")
random.seed(time.time()) # "this needs to be called before any other random functions"

randomnum = random.choice([5,2,3,0])
print(randomnum)
print(randomnum)
print(randomnum)

print(random.choice([5,2,3,0]))
print(random.choice([5,2,3,0]))
print(random.choice([5,2,3,0]))

print("*****************************************")
print("****************lists********************")
list = [1,2,3,4,5];
random.shuffle(list)
print("Reshuffled list : ",  list)

list = [ 'old', 786 , 2.23, 'john', 70.2 ]
tinylist = [66, 'peter']

print(list)          # Prints complete list
print(list[0])       # Prints first element of the list
print(list[-1])       # Prints last element of the list
print(list[1:3])     # Prints elements starting from 2nd till 3rd 
print(list[2:])      # Prints elements starting from 3rd element
print(tinylist * 3)  # Prints list 3 times
print(list + tinylist) # Prints concatenated lists
list[0] = 'new' #replaces the 0th part of list
print(list)

list = [1,2,3,4,5];
print(len(list)) #Gives the total length of the list.
print(max(list)) #Returns item from the list with max value.
print(min(list)) #Returns item from the list with min value.


print("*****************************************")
print("****************strings********************")
name = "Peter"
print(name)          # Prints complete string
print(name[0])       # Prints first character of the string
print(name[2:4])     # Prints characters starting from 3rd to 5th
print(name[2:])      # Prints string starting from 3rd character
print(name * 3)      # Prints string three times
print(name + " TEST") # Prints concatenated string
print("a" in name)
print("e" in name)
print("My name is %s and I am %d years old!" % (name, 23))  #format string

name = "maria"
print(name.capitalize()) #Capitalizes first letter of string
print(name.upper()) #Converts lowercase letters in string to uppercase
print(name.count("a")) #Counts how many times argument occurs in string
print(name.find("i")) #Determine if argument occurs in string and returns index of match
print(name.find("x")) #Determine if argument occurs in string and returns -1 if it doesn't exist
print(len(name)) #Returns the length of the string
print(name.split("a")) #Splits string according to delimiter argument


print("*****************************************")
print("****************conditional statements********************")
print("****************if statements********************")
positiveInteger = 2
if positiveInteger <= 5:
    print("Value of positiveInteger variable is less or equal to 5")
elif positiveInteger <= 10:
    print("Value of positiveInteger variable is larger than 5 but less or equal than 10")
else:
    print("Value of positiveInteger is larger than 10")
  
print("****************while statements********************")

count = 0
while (count < 5):
    print('The count is:', count)
    count = count + 1

print("Good bye!")

print("****************break statements********************")

for letter in 'Python':     
    if letter == 'h':
        break
    print('Current Letter :', letter)

print("****************continue statements********************")
for letter in 'Python':     
    if letter == 'h':
        continue
    print('Current Letter :', letter)
    
print("*****************************************")
print("****************functions********************") 

def printUpperCase(string):
    "This funtions turns the input argument string into upper case and prints the result"
    stringUpperCase = string.upper()
    print(stringUpperCase)

printUpperCase("i am now uppercase") #In this line we are calling the function previously defined

total = 0; # This is global variable.

# Function definition
def sum( arg1, arg2 ):
   # Add both input parameters and return their sum."
    total = arg1 + arg2; # Here total is local variable.
    print("Inside the function local total : ", total)
    return total;

# function invocation
returnedTotal = sum( 10, 20 );

print("Outside the function global total : ", total)
print("Value returned by function: ", returnedTotal)


print("*****************************************")
print("****************classes ********************") 
class Employee:
    'Common base class for all employees'   #documentation string
    empCount = 0

    def __init__(self, name, salary):  #constructor
        self.name = name
        self.salary = salary
        Employee.empCount += 1

    def displayCount(self):          #method
        print("Total Employee %d" % Employee.empCount)

    def displayEmployee(self):       #another method
        print("Name: ", self.name,  ", Salary: ", self.salary)
        
print(Employee.__doc__) # the documentation string can be accessed as Classname.__doc__

print("****************inheritance ********************") 
class Parent:        # define parent class
    parentAttr = 100
    def __init__(self):
        print("Calling parent constructor")

    def parentMethod(self):
        print('Calling parent method')

    def setAttr(self, attr):
        Parent.parentAttr = attr

    def getAttr(self):
        print("Parent attribute :", Parent.parentAttr)

class Child(Parent): # define child class which inherits from Parent
    def __init__(self):
        super().__init__() #Calling parent class constructor
        print("Calling child constructor")

    def childMethod(self):
        print('Calling child method')

c = Child()          # instance of child
print("------------")
c.childMethod()      # child calls its method
c.parentMethod()     # calls parent's method
c.setAttr(200)       # again call parent's method
c.getAttr()          # again call parent's method

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self): #Print method overloading
        return('Vector (%d, %d)' % (self.x, self.y))

    def __add__(self,other): #plus operator overloading
        return Vector(self.x + other.x, self.y + other.y)

a = Vector(4,2)
b = Vector(1,5)
c=a+b # we use the plus operator which has been overloaded by the vector class implementation
print(c) # the vector class also overloaded the print method