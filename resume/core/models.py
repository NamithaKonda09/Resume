from django.db import models


class Resume(models.Model):
    title = models.CharField(max_length=100)
    file = models.FileField(upload_to='resumepdfs/pdfs/')

    def __str__(self):
        return self.title
