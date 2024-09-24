from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
import json
import os

from assistant import chat_response


def index(request):
    template = loader.get_template("templates/ariassist/index.html")
    return HttpResponse(template.render({}, request))


@csrf_exempt
def chatbot(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_message = data.get("message")
        # Pass the user's message to your chatbot and get the response
        bot_message = chat_response(user_message)

        # Return the chatbot response as JSON
        return JsonResponse({"bot_message": bot_message})
