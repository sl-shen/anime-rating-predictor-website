import React, { useState, FormEvent } from 'react';
import axios from 'axios';
import '../global.css';

interface AnimeInput {
  title: string;
  summary: string;
}

interface AnimeRatingResponse {
  rating: number;
}

const AnimeRatingPredictor: React.FC = () => {
  const [title, setTitle] = useState<string>('');
  const [summary, setSummary] = useState<string>('');
  const [rating, setRating] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setRating(null);

    try {
      const response = await axios.post<AnimeRatingResponse>('http://localhost:8000/predict-rating/single', {
        title,
        summary
      } as AnimeInput);
      setRating(response.data.rating);
    } catch (err) {
      setError('An error occurred while predicting the rating. Please try again.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="anime-rating-predictor">
      <h2>Anime Rating Predictor</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="title">Anime Title</label>
          <input
            type="text"
            id="title"
            value={title}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setTitle(e.target.value)}
            required
          />
        </div>
        <div>
          <label htmlFor="summary">Anime Summary</label>
          <textarea
            id="summary"
            value={summary}
            onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setSummary(e.target.value)}
            rows={4}
            required
          ></textarea>
        </div>
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Predicting...' : 'Predict Rating'}
        </button>
      </form>
      {rating !== null && (
        <div className="result">
          <p>Predicted Rating: {rating.toFixed(2)}</p>
        </div>
      )}
      {error && (
        <div className="error">
          <p>{error}</p>
        </div>
      )}
    </div>
  );
};

export default AnimeRatingPredictor;