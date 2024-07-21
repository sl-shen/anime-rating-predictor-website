import React, { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import '../global.css';

interface LayoutProps {
    children: ReactNode;
  }
  
  const Layout: React.FC<LayoutProps> = ({ children }) => {
    return (
        <div>
          <nav>
            <ul>
              <li><Link to="/">Home</Link></li>
              <li><Link to="/new">New Anime Predictons</Link></li>
              <li><Link to="/single">Single Predictor</Link></li>
            </ul>
          </nav>
          <div className="page-content">
            {children}
          </div>
        </div>
    );
  };
  
  export default Layout;