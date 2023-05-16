import tkinter as tk
import project

root = tk.Tk()
root.title("hello to income predction program")
root.geometry("800x600")
root.resizable(True,True) # This removes the maximize button
root.configure(bg="light blue")
# Creating title label
title_label = tk.Label(root, text="Income Prediction", font=("Arial", 24),fg="black")
title_label.grid(row=0, column=2, columnspan=2, pady=10)

# Initializing Variables
age_text = tk.StringVar()
workclass_text = tk.StringVar()
education_text = tk.StringVar()
education_num_text = tk.StringVar()
marital_status_text = tk.StringVar()
occupation_text = tk.StringVar()
#race_text = tk.StringVar()
sex_text = tk.StringVar()
hours_per_week_text = tk.StringVar()
native_country_text = tk.StringVar()

# Creating Labels
label1 = tk.Label(root, text="age:")
label1.grid(row=1, column=1, sticky=tk.E, padx=2, pady=10)

label2 = tk.Label(root, text="workclass:")
label2.grid(row=2, column=1, sticky=tk.E, padx=2, pady=10)

label3 = tk.Label(root, text="education:")
label3.grid(row=3, column=1, sticky=tk.E, padx=2, pady=10)

label4 = tk.Label(root, text="education_num:")
label4.grid(row=4, column=1, sticky=tk.E, padx=2, pady=10)

label5 = tk.Label(root, text="marital_status:")
label5.grid(row=5, column=1, sticky=tk.E, padx=2, pady=10)

label6 = tk.Label(root, text="occupation:")
label6.grid(row=6, column=1, sticky=tk.E, padx=2, pady=10)

label8 = tk.Label(root, text="sex:")
label8.grid(row=7, column=1, sticky=tk.E, padx=2, pady=10)

label9 = tk.Label(root, text="hours_per_week:")
label9.grid(row=8, column=1, sticky=tk.E, padx=2, pady=10)

label10 = tk.Label(root, text="native_country:")
label10.grid(row=9, column=1, sticky=tk.E, padx=2, pady=10)

label_ans = tk.Label(root, text="prediction is : " , font=("Arial",15))
label_ans.grid(row=5, column=4, sticky=tk.E, padx=2, pady=10)

# Creating Text Fields
entry1 = tk.Entry(root, textvariable=age_text)
entry1.grid(row=1, column=2, padx=2, pady=10, sticky=tk.W)

entry2 = tk.Entry(root, textvariable=workclass_text)
entry2.grid(row=2, column=2, padx=2, pady=10, sticky=tk.W)

entry3 = tk.Entry(root, textvariable=education_text)
entry3.grid(row=3, column=2, padx=2, pady=10, sticky=tk.W)

entry4 = tk.Entry(root, textvariable=education_num_text)
entry4.grid(row=4, column=2, padx=2, pady=10, sticky=tk.W)

entry5 = tk.Entry(root, textvariable=marital_status_text)
entry5.grid(row=5, column=2, padx=2, pady=10, sticky=tk.W)

entry6 = tk.Entry(root, textvariable=occupation_text)
entry6.grid(row=6, column=2, padx=2, pady=10, sticky=tk.W)

entry8 = tk.Entry(root, textvariable=sex_text)
entry8.grid(row=7, column=2, padx=2, pady=10, sticky=tk.W)

entry9 = tk.Entry(root, textvariable=hours_per_week_text)
entry9.grid(row=8, column=2, padx=2, pady=10, sticky=tk.W)

entry10 = tk.Entry(root, textvariable=native_country_text)
entry10.grid(row=9, column=2, padx=2, pady=10, sticky=tk.W)

# Creating Button Function
def Predict():
      ourData=[age_text.get(),workclass_text.get(),education_text.get(),education_num_text.get(),marital_status_text.get(),
      occupation_text.get(),sex_text.get(),hours_per_week_text.get(),native_country_text.get()]
      #ourData = [20,"Private","Bachelors",13,"Divorced","Sales","Male",30,"Mexico"]
      final_predict = project.runIncomePrediction(ourData)
      answer ="prediction is : "+ final_predict
      label_ans = tk.Label(root, text=answer ,font=("Arial", 15))
      label_ans.grid(row=5, column=4, sticky=tk.E, padx=2, pady=10)
# Creating Button 
button = tk.Button(root, text="predict", command=Predict)
button.grid(row=11, column=1, columnspan=2, pady=2)

root.mainloop()