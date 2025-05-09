from django import forms
from .models import PredictionInputs

class CustomLoginForm(forms.Form):
    email = forms.EmailField(label="Email", widget=forms.EmailInput(attrs={'class': 'form-control'}))
    password = forms.CharField(label="Password", widget=forms.PasswordInput(attrs={'class': 'form-control'}))

class PredictionForm(forms.ModelForm):
    class Meta:
        model = PredictionInputs
        exclude = ['user']

        widgets = {
            'volatile_acidity': forms.NumberInput(attrs={'class': 'form-control'}),
            'citric_acid': forms.NumberInput(attrs={'class': 'form-control'}),
            'density': forms.NumberInput(attrs={'class': 'form-control'}),
            'pH': forms.NumberInput(attrs={'class': 'form-control'}),
            'alcohol': forms.NumberInput(attrs={'class': 'form-control'}),
            'residual_sugar': forms.NumberInput(attrs={'class': 'form-control'}),
            'chloride': forms.NumberInput(attrs={'class': 'form-control'}),
            'sulphates': forms.NumberInput(attrs={'class': 'form-control'}),
            'fixed_acidity': forms.NumberInput(attrs={'class': 'form-control'}),
            'free_sulfur_dioxide': forms.NumberInput(attrs={'class': 'form-control'}),
            'total_sulfur_dioxide': forms.NumberInput(attrs={'class': 'form-control'}),
        }