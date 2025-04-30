from django.db import models

# class AnonymousUser(models.Model):
#     uuid = models.CharField(max_length=255)
#     name = models.CharField(max_length=255)
#     image_url = models.TextField(default="Unknown")
#     created_at = models.DateTimeField(auto_now_add=True)

#     def __str__(self):
#         return f"{self.uuid}"

# class Post(models.Model):
#     user = models.ForeignKey(AnonymousUser, on_delete=models.CASCADE, related_name="posts")
#     caption = models.TextField()
#     total_duration = models.IntegerField(default=0) 
#     device_usage = models.CharField(max_length=255, default="Unknown")
#     likes_count = models.IntegerField(default=0)
#     comments_count = models.IntegerField(default=0)
#     is_like = models.BooleanField(default=False)
#     created_at = models.DateTimeField(auto_now_add=True)

#     def __str__(self):
#         return f"{self.user.name}"