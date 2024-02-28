from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def getImageList(request):
    id = request.GET.get('id')
    name = request.GET.get('name')
    
    return JsonResponse({name: id})
