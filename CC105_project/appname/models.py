from django.db import models
from django.contrib.auth.models import User

class PredictionInputs(models.Model):
    id = models.AutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    volatile_acidity = models.FloatField()
    citric_acid = models.FloatField()
    density = models.FloatField()
    pH = models.FloatField()
    alcohol = models.FloatField()
    residual_sugar = models.FloatField()
    chloride = models.FloatField()
    sulphates = models.FloatField()
    fixed_acidity = models.FloatField()
    free_sulfur_dioxide = models.FloatField()
    total_sulfur_dioxide = models.FloatField()

    def __str__(self):
        return f"{self.id}, {self.user}, {self.volatile_acidity}, {self.citric_acid}, {self.density}, {self.pH}, {self.alcohol}, {self.residual_sugar}, {self.chloride}, {self.sulphates}, {self.fixed_acidity}, {self.free_sulfur_dioxide}, {self.total_sulfur_dioxide}"