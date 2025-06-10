from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from screen_time_ml.utils.classification_usage import predict_from_api

@api_view(['POST'])
def intensity_classification(request):
    try:
        app_usage_minutes = request.data.get('app_usage_minutes')
        screen_time_hours = request.data.get('screen_time_hours')
        app_frequency = request.data.get('app_frequency')
        age = request.data.get('age')
        print("Received data:", request.data)

        if not all([app_usage_minutes, screen_time_hours, app_frequency, age]):
            return Response({
                "status": status.HTTP_400_BAD_REQUEST,
                "message": "All fields are required."
            }, status=status.HTTP_400_BAD_REQUEST)

        prediction, confidence, insights = predict_from_api(app_usage_minutes, screen_time_hours, app_frequency, age)
        confidence_percent = int(round(float(confidence) * 100))
        
        return Response({
            "status": status.HTTP_200_OK,
            "message": "Classification successful",
            "data": {
                "classification": prediction,
                "confidence": confidence_percent,
                "insights": insights
            }
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({
            "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)