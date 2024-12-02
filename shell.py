#Code is semi aided by GeeksForGeeks (included in resources for this project) and CodePulse (also in class resources)

import Kobra

while True:
    text = input('Kobra > ')
    result, error = Kobra.run('<stdin>', text)

    if error:
        print(error.as_string())
    else:
        # Access and print the value from the RTResult object
        print(result.value)
