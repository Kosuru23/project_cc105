from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from .forms import CustomLoginForm
from django.contrib.auth.decorators import login_required
from .forms import PredictionForm
from scipy.special import boxcox
import pandas as pd
import xgboost as xgb
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from io import BytesIO
import base64

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, 'xgb_model.pkl')
lambda_path = os.path.join(BASE_DIR, 'lambda_path.pkl')
fitted_path = os.path.join(BASE_DIR, 'fitted_lambda_path.pkl')
yeo_chloride = os.path.join(BASE_DIR, 'pt.pkl')
sqrt_acid_fix = os.path.join(BASE_DIR, 'sqrt_fixed_acidity.pkl')
ftsd = os.path.join(BASE_DIR, 'ftsd.pkl')
ltsd = os.path.join(BASE_DIR, 'ltsd.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(lambda_path, 'rb') as f:
    sugar_lambda = pickle.load(f)
with open(fitted_path, 'rb') as f:
    sulphates_lambda = pickle.load(f)
with open(yeo_chloride, 'rb') as f:
    pt = pickle.load(f)
with open(sqrt_acid_fix, 'rb') as f:
    acid_fix = pickle.load(f)
with open(ftsd, 'rb') as f:
    ftsd = pickle.load(f)
with open(ltsd, 'rb') as f:
    ltsd = pickle.load(f)

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
            
            residual_sugar = input_vector[5]
            chloride = input_vector[6]
            sulphates = input_vector[7]
            fixed_acidity = input_vector[8]
            free_sulfur = input_vector[9]
            total_sulfur = input_vector[10]

            if residual_sugar <= 0:
                form.add_error(None, 'Residual sugar must be greater than 0 for Box-Cox transformation.')
                return render(request, 'prediction.html', {'form': form})
            else:
                residual_sugar_array = np.array([residual_sugar], dtype=np.float64)

                input_vector[5] = boxcox(residual_sugar_array, float(sugar_lambda))[0]

            input_df = pd.DataFrame([[chloride]], columns=['chlorides'])
            input_vector[6] = pt.transform(input_df)
            
            if sulphates <= 0:
                form.add_error(None, 'Sulphates must be greater than 0 for Box-Cox transformation.')
                return render(request, 'prediction.html', {'form': form})
            else:
                sulphates_array = np.array([sulphates], dtype=np.float64)
                input_vector[7] = boxcox(sulphates_array, float(sulphates_lambda))[0]

            new_data = pd.DataFrame([[fixed_acidity]], columns=['fixed_acidity'])
            input_vector[8] = acid_fix.transform(new_data)
            
            new_data = pd.DataFrame([[free_sulfur]], columns=['free_sulfur'])
            input_vector[9] = ftsd.transform(new_data)

            new_data = pd.DataFrame([[total_sulfur]], columns=['total_sulfur'])
            input_vector[10] = ltsd.transform(new_data)

            def extract_value(x):
                if isinstance(x, pd.DataFrame):
                    return float(x.values.flatten()[0])
                elif isinstance(x, np.ndarray):
                    return float(x.flatten()[0])
                else:
                    return float(x)

            cleaned = [extract_value(x) for x in input_vector]
        
            prediction = model.predict([cleaned])[0]
            print("I predict:", prediction)

            return render(request, 'prediction.html', {'form': form, 'prediction': prediction})
        else:
            print("Form errors:", form.errors) 
    else:
        form = PredictionForm()

    return render(request, 'prediction.html', {'form': form})

def dashboard(request):

    def plot_to_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode()
        buf.close()
        plt.close(fig)
        return image_base64
    
    def bin_quality(q):
        if q <= 2:
            return 0  
        elif q <= 4:
            return 1  
        else:
            return 2 

    csv_path = os.path.join(BASE_DIR, 'static', 'dashboard', 'WineQT.csv')
    df = pd.read_csv(csv_path)

    df['residual_sugar_boxcox'] = df['residual_sugar'].apply(
        lambda x: boxcox([x], float(sugar_lambda))[0] if x > 0 else 0)

    df['yeo_chloride'] = pt.transform(df[['chlorides']])

    df['sulphates_boxcox'] = df['sulphates'].apply(
        lambda x: boxcox([x], float(sulphates_lambda))[0] if x > 0 else 0)

    df['sqrt_fixed_acidity'] = acid_fix.transform(df[['fixed_acidity']])

    df['log_free_sulfur_dioxide'] = ftsd.transform(df[['free_sulfur_dioxide']])
    df['log_total_sulfur_dioxide'] = ltsd.transform(df[['total_sulfur_dioxide']])
    
    df = df.drop('residual_sugar', axis=1)
    df = df.drop('chlorides', axis=1)
    df = df.drop('fixed_acidity', axis=1)
    df = df.drop('sulphates', axis=1)
    df = df.drop('free_sulfur_dioxide', axis=1)
    df = df.drop('total_sulfur_dioxide', axis=1)

    features=df.drop(columns=['quality']).columns.values
    targets='quality'
    target=df["quality"]
    target = target - 3
    train_df,test_df=train_test_split(df,test_size=0.2, random_state=42,stratify=df.quality)
    X_train=train_df[features]
    y_train_transformed=train_df[targets]
    X_test=test_df[features]
    y_test_transformed=test_df[targets]
    y_train = y_train_transformed - 3
    y_test = y_test_transformed - 3
    y_train_binned = y_train.apply(bin_quality)
    y_test_binned = y_test.apply(bin_quality)

    y_pred = model.predict(X_test)

    # Accuracy and Confusion Matrix
    acc = accuracy_score(y_test_binned, y_pred)
    cm = confusion_matrix(y_test_binned, y_pred)

    # Charts
    fig1, ax1 = plt.subplots()
    y_test_binned.value_counts().sort_index().plot(kind='bar', ax=ax1, title='Target Distribution')
    bar_chart = plot_to_base64(fig1)

    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('Confusion Matrix')
    confusion_plot = plot_to_base64(fig2)

    # Summary of raw features before transform
    summary_html = df[features].describe().to_html(classes='table table-striped')

    context = {
        'num_records': len(df),
        'summary_html': summary_html,
        'accuracy': f"{acc:.2f}",
        'bar_chart': bar_chart,
        'confusion_plot': confusion_plot,
    }

    return render(request, 'dashboard.html', context)