import React, { useState, useEffect } from 'react';
import axios from 'axios';
import '../global.css';

const DEFAULT_IMAGE_URL = '/public/default.jpg';

interface Anime {
  id: string;
  title: string;
  image_url: string;
  summary: string;
  year: number;
  month: number;
  rating: number;
}

const AnimeCard: React.FC<{ anime: Anime }> = ({ anime }) => {
  const [imgSrc, setImgSrc] = useState(anime.image_url || DEFAULT_IMAGE_URL);

  const handleImageError = () => {
    if (imgSrc !== DEFAULT_IMAGE_URL) {
      setImgSrc(DEFAULT_IMAGE_URL);
    }
  };

  return (
    <div className="anime-card">
      <a href={`https://bangumi.tv/subject/${anime.id}`} target="_blank" rel="noopener noreferrer" className="anime-image-link">
        <img src={anime.image_url} alt={anime.title} className="anime-image" onError={handleImageError} />
      </a>
      <div className="anime-info">
        <a href={`https://bangumi.tv/subject/${anime.id}`} target="_blank" rel="noopener noreferrer" className="anime-title">
          {anime.title}
        </a>
        <p className="anime-rating">
          {`预测评分: ${anime.rating.toFixed(2)}`}
        </p>
      </div>
    </div>
  );
};

const AnimeList: React.FC = () => {
  const [animeList, setAnimeList] = useState<Anime[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnime = async () => {
      try {
        const response = await axios.get<Anime[]>('http://localhost:8000/api/current-anime');
        const sortedAnime = response.data.sort((a, b) => {
          if (a.year !== b.year) return a.year - b.year;
          return a.month - b.month;
        });
        setAnimeList(sortedAnime);
      } catch (error) {
        console.error('Error fetching anime data:', error);
        setError('Failed to fetch anime data. Please try again later.');
      }
    };

    fetchAnime();
  }, []);

  const groupedAnime = animeList.reduce((acc, anime) => {
    const key = `${anime.year}-${anime.month.toString().padStart(2, '0')}`;
    if (!acc[key]) acc[key] = [];
    acc[key].push(anime);
    return acc;
  }, {} as Record<string, Anime[]>);

  if (error) {
    return <div className="error-message">{error}</div>;
  }

  return (
    <div className="anime-list-container">
      {Object.entries(groupedAnime).map(([yearMonth, animes]) => (
        <React.Fragment key={yearMonth}>
          <h2 className="anime-year-month">{yearMonth}</h2>
          <div className="anime-grid">
            {animes.map((anime) => (
              <AnimeCard key={anime.id} anime={anime} />
            ))}
          </div>
        </React.Fragment>
      ))}
    </div>
  );
};

export default AnimeList;