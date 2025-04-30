from rest_framework import serializers
from django.contrib.auth.models import User
from . import models

# class UserSerializer(serializers.ModelSerializer):
#     is_superuser = serializers.BooleanField(default=False, required=False)

#     class Meta(object):
#         model = User
#         fields = ['id', 'username', 'email', 'password', 'is_superuser']
# class PostSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = models.Post
#         fields = ['id', 'user', 'caption', 'total_duration', 'device_usage', 'likes_count', 'comments_count', 'is_like', 'created_at']