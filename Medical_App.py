from tkinter import *
import joblib
import pandas as pd
import warnings
import sklearn
import sklearn.ensemble._weight_boosting


warnings.filterwarnings('ignore')

def write():
    feature = [[medicine.get(), float(price.get()), area.get(), speciality.get(),
                dr_class.get(), float(exam_price.get()), clinic_hos.get()]]
    columns = ['medicine', 'price', 'area', 'speciality', 'dr_class', 'exam_price', 'clinic_hos']

    df = pd.DataFrame(feature, columns=columns)

    h_encoder = joblib.load('h_encoder.pkl')
    scaler = joblib.load('scaler.pkl')

    df['dr_class'] = df['dr_class'].map({'b': 0, 'a': 1})
    df['clinic_hos'] = df['clinic_hos'].map({'clinic': 0, 'hospital': 1})

    numeric = df.select_dtypes(include=["int64", "float64"])
    catg = df.select_dtypes(include=["object"])

    features_cat = h_encoder.transform(df[catg.columns])
    encoded_features_df = pd.DataFrame(features_cat, columns=h_encoder.get_feature_names_out(catg.columns), index=df.index)
    row_processed = pd.concat([df[numeric.columns], encoded_features_df], axis=1)

    row_processed[["price", "exam_price"]] = scaler.transform(row_processed[["price", "exam_price"]])

    loaded_model = joblib.load('loaded_clf.pkl')
    pred = loaded_model.predict(row_processed)

    pred = "Will Write" if str(pred[0]) == "1" else "Will Not Write"

    t1.delete(0.0, END)
    t1.insert(0.0, pred)

# Initialize main window
wind = Tk()
wind.title("Medicine Prediction")
wind.geometry("1000x500")  # Increase window size
wind.configure(bg="#2C3E50")  # Dark Blue Background

# Load and place the image
image = PhotoImage(file="medical_rep.png")  # Load the image
image_label = Label(wind, image=image, bg="#2C3E50")  # Set the background color to match
image_label.place(relx=0.5, rely=0.4, anchor=CENTER)  # Position the image at the center but lower (rely=0.2)

# Center Frame for Input Fields
input_frame = Frame(wind, bg="#34495E")
input_frame.place(relx=0.5, rely=0.5, anchor=CENTER)  # Center the frame in the window
input_frame.pack_propagate(False)  # Prevent the frame from shrinking to fit widgets
input_frame.config(width=900, height=250)  # Set a fixed width and height for the frame

# Labels for the first row (3 labels)
labels_top = ["Medicine", "Medicine Price", "Area"]
for i, label in enumerate(labels_top):
    Label(input_frame, text=label, bg="#34495E", fg="white", font=('Helvetica', 12)).grid(row=0, column=i, padx=10, pady=10)

# Entries for the first row
medicine = StringVar()
Entry(input_frame, textvariable=medicine, font=('Helvetica', 12)).grid(row=1, column=0, padx=10, pady=10)

price = StringVar()
Entry(input_frame, textvariable=price, font=('Helvetica', 12)).grid(row=1, column=1, padx=10, pady=10)

area = StringVar()
Entry(input_frame, textvariable=area, font=('Helvetica', 12)).grid(row=1, column=2, padx=10, pady=10)

# Labels for the second row (3 labels)
labels_middle = ["Doctor Speciality", "Doctor Class", "Examination Price"]
for i, label in enumerate(labels_middle):
    Label(input_frame, text=label, bg="#34495E", fg="white", font=('Helvetica', 12)).grid(row=2, column=i, padx=10, pady=10)

# Entries for the second row
speciality = StringVar()
Entry(input_frame, textvariable=speciality, font=('Helvetica', 12)).grid(row=3, column=0, padx=10, pady=10)

dr_class = StringVar()
Entry(input_frame, textvariable=dr_class, font=('Helvetica', 12)).grid(row=3, column=1, padx=10, pady=10)

exam_price = StringVar()
Entry(input_frame, textvariable=exam_price, font=('Helvetica', 12)).grid(row=3, column=2, padx=10, pady=10)

# Label and Entry for Clinic Or Hospital (centered)
Label(input_frame, text="Clinic Or Hospital", bg="#34495E", fg="white", font=('Helvetica', 12)).grid(row=4, column=1, padx=10, pady=10)

clinic_hos = StringVar()
Entry(input_frame, textvariable=clinic_hos, font=('Helvetica', 12)).grid(row=5, column=1, padx=10, pady=10)

# Button
b1 = Button(wind, text="Predict", command=write, bg="#1ABC9C", fg="white", font=('Helvetica', 12))
b1.place(relx=0.5, rely=0.82, anchor=CENTER)  # Moved up to 0.82

# Output Text
t1 = Text(wind, height=1, width=20, bg="#ECF0F1", fg="#2C3E50", font=('Helvetica', 12))
t1.place(relx=0.5, rely=0.9, anchor=CENTER)  # Adjust output text box position lower

# Run the main loop
wind.mainloop()
