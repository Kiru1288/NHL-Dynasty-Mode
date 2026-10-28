import React from 'react';
import ReactDOM from 'react-dom/client';
import './styles/game.css';

import { applyFluidUiScale } from './utils/fluidUiScale';
import { applyGraphicsQualityClass } from './utils/graphicsQuality';
import { installScreenScopes } from './styles/screenScopes';
import App from './App';
import reportWebVitals from './reportWebVitals';

// Cartoon arcade look (game.css: html.cartoon-ui). Remove to fall back to the flat skin.
document.documentElement.classList.add("cartoon-ui");
applyFluidUiScale();
applyGraphicsQualityClass();
installScreenScopes();

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

reportWebVitals();
