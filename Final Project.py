# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 12:44:07 2023

@author: wmcdo
"""

from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np

DeepFace.stream(db_path='C:/Users/wmcdo/OneDrive/Fall 2023/M E 369P/Project/databases', time_threshold=5, frame_threshold=1)