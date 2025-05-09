from django.urls import path
from . import views
from django.contrib.auth.views import LogoutView

urlpatterns = [
    path('', views.home, name='home'),  
    path('login/', views.login_view, name='login'),
    path('logout/', LogoutView.as_view(next_page=''), name='logout'),  
    path('register/', views.register, name='register'),
    path('start_prediction/', views.start_prediction, name='start_prediction'),
    path('save-input/', views.save_input, name='save_input'),
    path('dashboard/', views.dashboard, name='dashboard'),
]