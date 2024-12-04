#Code is semi aided by GeeksForGeeks (included in resources for this project) and CodePulse (also in class resources)
#By Evan Valenti
#For CSC-333-101 Final Project: Create a Programming Language

import Kobra


interpreter = Kobra.Interpreter()

while True:
   
    text = input("Kobra > ")
    
    if text.strip().lower() == "exit":
        break  
    
    
    result, error = Kobra.run('<stdin>', text, interpreter)
    
    if error:
        print(error)
    elif result:
       
        print(result.value if hasattr(result, 'value') else result)




