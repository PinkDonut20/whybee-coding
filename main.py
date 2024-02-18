import telebot
import torch
from telebot import types
from diffusers import StableDiffusionPipeline

token='6754666483:AAEFArY_vhZg6T9Qn0evF4xPCnXaF192q2A'
bot=telebot.TeleBot(token)

repo_path = r"C:\Users\Николай\stable-diffusion\stable-diffusion-2"

pipe = StableDiffusionPipeline.from_pretrained(repo_path, torch_dtype=torch.float32)
pipe = pipe.to("cuda")

@bot.message_handler(commands=['start'])
def start_message(message):
	markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
	itembtn1 = types.KeyboardButton('Сгенерировать картинку')
	markup.add(itembtn1)
	bot.send_message(message.chat.id, 'Выбери то, что нужно!', reply_markup=markup)
    
@bot.message_handler(content_types='text')
def message_reply(message):
	if message.text == 'Сгенерировать картинку':
		bot.send_message(message.chat.id,'Введите то, что Вы хотите сгенерировать. Нейросеть работает только на английском языке. No NSFW content!', reply_markup=types.ReplyKeyboardRemove())
	else:
		prompt = message.text
		image = pipe(prompt).images[0]
		bot.send_photo(message.chat.id, image)
		markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
		itembtn1 = types.KeyboardButton('Сгенерировать картинку')
		markup.add(itembtn1)
		bot.send_message(message.chat.id, 'Выбери то, что нужно!', reply_markup=markup)
        
bot.infinity_polling()