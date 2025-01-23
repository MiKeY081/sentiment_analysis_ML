import { useState } from 'react';
import { SearchBar } from '@/components/SearchBar';
import { SentimentChart } from '@/components/SentimentChart';
import { TweetList } from '@/components/TweetList';
import { SentimentResults } from '@/types';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useToast } from '@/hooks/use-toast';
import axios from 'axios';



export default function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<SentimentResults | null>(null);
  const { toast } = useToast();

  const handleSearch = async (query: string) => {
    setIsLoading(true);
    setResults(null);
    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', {
        query, // Send keyword/hashtag to the backend
      });
      if(response.data){
      setResults(response.data);
    }else{
      toast({ description: 'No results found.', variant: 'warning' });
    }

    } catch (error) {
      console.error('Error fetching sentiment analysis:', error);
      toast({ description: 'Failed to analyze sentiment.', variant: 'destructive' });
    } finally {
      setIsLoading(false);
    }
  };
  

  return (
    <div className="min-h-screen bg-background p-8 min-w-screen ">
      <div className="max-w-6xl mx-auto space-y-8">
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold">Tweet Sentiment Analysis</h1>
          <p className="text-muted-foreground">
            Enter a keyword or hashtag to analyze the sentiment of related tweets
          </p>
        </div>

        <div className="flex justify-center">
          <SearchBar onSearch={handleSearch} isLoading={isLoading} />
        </div>

        {results && (
          <div className="space-y-8">
            <div className="bg-card p-6 rounded-lg shadow-sm">
              <h2 className="text-2xl font-bold mb-4">Sentiment Distribution</h2>
              <SentimentChart distribution={results.distribution} />
            </div>

            <Tabs defaultValue="positive" className="w-full">
              <TabsList className="grid w-full grid-cols-3 gap-3">
                <TabsTrigger value="positive">Positive</TabsTrigger>
                <TabsTrigger value="neutral">Neutral</TabsTrigger>
                <TabsTrigger value="negative">Negative</TabsTrigger>
              </TabsList>
              <TabsContent value="positive">
                <TweetList tweets={results.tweets} sentiment="positive" />
              </TabsContent>
              <TabsContent value="neutral">
                <TweetList tweets={results.tweets} sentiment="neutral" />
              </TabsContent>
              <TabsContent value="negative">
                <TweetList tweets={results.tweets} sentiment="negative" />
              </TabsContent>
            </Tabs>
          </div>
        )}
      </div>
    </div>
  );
}