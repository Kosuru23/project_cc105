from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from .forms import CustomLoginForm
from django.contrib.auth.decorators import login_required
from .forms import PredictionForm
from scipy.special import boxcox
import pickle
import os

model_path = os.path.join(os.path.dirname(__file__), 'xgb_models.pkl')
rs_boxcox = os.path.join(os.path.dirname(__file__), 'lambda_path.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(rs_boxcox, 'rb') as f:
    sugar = pickle.load(f)


# Home page view
def home(request):
    return render(request, 'home.html')

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save() 
            login(request, user)  
            return redirect('home')  
    else:
        form = UserCreationForm()

    return render(request, 'register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = CustomLoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']

            user = authenticate(request, username=email, password=password)

            if user is not None:
                login(request, user)  # Log the user in
                return redirect('home')  # Redirect to home page after successful login
            else:
                form.add_error(None, 'Invalid email or password')  # Show error if authentication fails
    else:
        form = CustomLoginForm()

    return render(request, 'login.html', {'form': form})

@login_required
def start_prediction(request):
    return render(request, 'prediction.html')

def log_out(request):
    logout(request)  # Log the user out
    return redirect('home')  # Redirect to home page after logging out

def save_input(request):
    prediction = None

    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            print("Form is valid")  
            user_input = form.save(commit=False)
            user_input.user = request.user
            user_input.save()
            
            input_vector = [
                user_input.volatile_acidity,
                user_input.citric_acid,
                user_input.density,
                user_input.pH,
                user_input.alcohol,
                user_input.residual_sugar,
                user_input.chloride,
                user_input.sulphates,
                user_input.fixed_acidity,
                user_input.free_sulfur_dioxide,
                user_input.total_sulfur_dioxide
            ]

            input_vector[5] = boxcox(input_vector[5], lmbda=sugar) 
            prediction = model.predict([input_vector])[0]

            return render(request, 'prediction.html', {'form': form, 'prediction': prediction})
        else:
            print("Form errors:", form.errors)  # Debug
    else:
        form = PredictionForm()

    return render(request, 'prediction.html', {'form': form})