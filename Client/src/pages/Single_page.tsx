import { Link } from 'react-router-dom';
import Layout from '../component/Layout';
import '../global.css';

const single = () => {
    return (
        <Layout>
            <div>
                <h1>Anime Rating predictor</h1>

                <Link to="/">Go back to Home</Link>
            </div>
        </Layout>
        
    )
}

export default single;