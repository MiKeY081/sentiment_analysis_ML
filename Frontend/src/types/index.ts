export interface Tweet {
  id: string;
  text: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  score: number;
}

export interface SentimentResults {
  distribution: 'positive' | 'negative' | 'neutral';
  tweets: Tweet[];
}