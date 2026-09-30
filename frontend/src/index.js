import React from 'react';
import ReactDOM from 'react-dom/client';
import './styles/global.css';
import { applyFluidUiScale } from './utils/fluidUiScale';
import { applyGraphicsQualityClass } from './utils/graphicsQuality';
import App from './App';
import reportWebVitals from './reportWebVitals';

applyFluidUiScale();
applyGraphicsQualityClass();

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

reportWebVitals();
