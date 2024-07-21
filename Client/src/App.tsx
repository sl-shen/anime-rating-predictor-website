import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Layout from './component/Layout';
import Single from './pages/Single_page';
import New from './pages/New_Anime_page';
import './global.css';

// Home component
const Home: React.FC = () => {
  return (
    <Layout>
      <div>
        <h1>Welcome to Anime Rating Predictor <br /> ------------------------kksk!------------------------- </h1>
        
      </div>
    </Layout>
   
  );
};

// App component
const App: React.FC = () => { 
  return (
      <Router>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/single" element={<Single />} />
            <Route path="/new" element={<New />} />
          </Routes>
      </Router>
  );
};

export default App;