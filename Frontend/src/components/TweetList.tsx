import { Tweet } from '@/types';
import { Card } from '@/components/ui/card';
import { MessageCircle, ThumbsUp, ThumbsDown, Minus } from 'lucide-react';

interface TweetListProps {
  tweets: Tweet[];
  sentiment: 'positive' | 'negative' | 'neutral';
}

export function TweetList({ tweets, sentiment }: TweetListProps) {
  const sentimentIcon = {
    positive: <ThumbsUp className="h-4 w-4 text-green-500" />,
    negative: <ThumbsDown className="h-4 w-4 text-red-500" />,
    neutral: <Minus className="h-4 w-4 text-gray-500" />,
  };

  const filteredTweets = tweets.filter((tweet) => tweet.sentiment === sentiment);

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold flex items-center gap-2">
        {sentimentIcon[sentiment]}
        {sentiment.charAt(0).toUpperCase() + sentiment.slice(1)} Tweets
      </h2>
      <div className="grid gap-4">
        {filteredTweets.map((tweet) => (
          <Card key={tweet.id} className="p-4">
            <div className="flex items-start gap-3">
              <MessageCircle className="h-5 w-5 mt-1 text-muted-foreground" />
              <div>
                <p className="text-sm text-foreground">{tweet.text}</p>
                <p className="text-xs text-muted-foreground mt-2">
                  Sentiment Score: {tweet.score.toFixed(2)}
                </p>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}