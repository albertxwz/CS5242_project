from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import time

@csrf_exempt
def generateImage(request):
    data = json.loads(request.body)
    print(data)
    code = data['code']
    time.sleep(3)
    return JsonResponse({'status': 200, 'data': code})
