from diffusers import StableDiffusionPipeline
from torch import autocast
import torch
from PIL import ImageTk, Image
import io
import customtkinter as ctk
import tkinter as tk

# Create the app
app = tk.Tk()
app.geometry("532x622")
app.title("Stable Buds")
ctk.set_appearance_mode("dark")

#Text box for giving the input text
prompt = ctk.CTkEntry(
     app,
     height=40,
     width=512,
     placeholder_text=("Arial",20),
     text_color="black",
     fg_color="white")
prompt.place(x=10,y=10)


#placeholder for image output
lmain = ctk.CTkLabel(master=app,height=512,width=512)
lmain.place(x=10,y=120)

#Getting authorisation token 
import requests

API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
API_TOKEN="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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
	response = requests.post(
        API_URL, 
        headers=headers, 
        json=payload
     )
	return response.content

def get_image_from_prompt( prompt ) :
     image_bytes = query( {"inputs": prompt,} )
     image = Image.open(io.BytesIO(image_bytes))
     return image

def generate():
     image = get_image_from_prompt( prompt.get() )

     img = ImageTk.PhotoImage(image)
     image.save('generatedimage.png')
     lmain.configure(image=img)

#Button to generate the image from the given text
trigger=ctk.CTkButton(
     master=app,
     height=40,
     width=110,
     font=("Arial",30),
     text_color="white",
     fg_color="black",
     command=generate)

trigger.configure(text="GENERATE")
trigger.place(x=166,y=80)

app.mainloop()