from django.db import models

class AboutPage(models.Model):
    title = models.CharField(max_length=255, default="About PsychGen-Africa")
    introduction = models.TextField()
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title


class Mission(models.Model):
    about_page = models.ForeignKey(AboutPage, on_delete=models.CASCADE, related_name='mission')
    title = models.CharField(max_length=255, default="Our Mission")
    content = models.TextField()

    def __str__(self):
        return f"Mission - {self.about_page.title}"


class Objective(models.Model):
    about_page = models.ForeignKey(AboutPage, on_delete=models.CASCADE, related_name='objectives')
    content = models.CharField(max_length=255)

    def __str__(self):
        return f"Objective - {self.content[:50]}"


class KeyFeature(models.Model):
    about_page = models.ForeignKey(AboutPage, on_delete=models.CASCADE, related_name='key_features')
    title = models.CharField(max_length=255)
    content = models.TextField()

    def __str__(self):
        return f"Key Feature - {self.title}"


class TechnologyDevelopment(models.Model):
    about_page = models.ForeignKey(AboutPage, on_delete=models.CASCADE, related_name='technology')
    title = models.CharField(max_length=255, default="Technology and Development")
    content = models.TextField()

    def __str__(self):
        return f"Technology - {self.about_page.title}"


class Vision(models.Model):
    about_page = models.ForeignKey(AboutPage, on_delete=models.CASCADE, related_name='vision')
    title = models.CharField(max_length=255, default="Our Vision")
    content = models.TextField()

    def __str__(self):
        return f"Vision - {self.about_page.title}"
