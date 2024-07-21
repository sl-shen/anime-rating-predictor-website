import { Link } from 'react-router-dom';
import Layout from '../component/Layout';
import '../global.css';
import AnimeList from '../component/AnimeList';

const new_anime = () => {
    return (
        <Layout>
            <div>
                <h1>New Anime predictons</h1>
                <AnimeList/>
                <Link to="/">Go back to Home</Link>
            </div>
        </Layout>
        
    )
}

export default new_anime;