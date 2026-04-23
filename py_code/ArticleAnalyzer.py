import re
import pandas as pd
from transformers import pipeline
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import time


class ArticleAnalyzer(ABC):
    entity_dict = {
        'drivers': {
            'Verstappen': {'team': 'Red Bull', 'role_context': 'Red Bull world champion',
                           'aliases': ['Max', 'Max Verstappen']},
            'Hadjar': {'team': 'Red Bull', 'role_context': 'Red Bull rookie driver',
                       'aliases': ['Isack', 'Isack Hadjar']},
            'Antonelli': {'team': 'Mercedes', 'role_context': 'Mercedes young driver',
                          'aliases': ['Kimi', 'Kimi Antonelli', 'Andrea Kimi Antonelli']},
            'Russell': {'team': 'Mercedes', 'role_context': 'Mercedes lead driver',
                        'aliases': ['George', 'George Russell']},
            'Leclerc': {'team': 'Ferrari', 'role_context': 'Ferrari lead driver',
                        'aliases': ['Charles', 'Charles Leclerc']},
            'Hamilton': {'team': 'Ferrari', 'role_context': 'Ferrari seven-time champion',
                         'aliases': ['Lewis', 'Lewis Hamilton', 'Sir Lewis']},
            'Norris': {'team': 'McLaren', 'role_context': 'McLaren world champion',
                       'aliases': ['Lando', 'Lando Norris']},
            'Piastri': {'team': 'McLaren', 'role_context': 'McLaren race winner',
                        'aliases': ['Oscar', 'Oscar Piastri']},
            'Gasly': {'team': 'Alpine', 'role_context': 'Alpine lead driver', 'aliases': ['Pierre', 'Pierre Gasly']},
            'Colapinto': {'team': 'Alpine', 'role_context': 'Alpine rookie driver',
                          'aliases': ['Franco', 'Franco Colapinto']},
            'Lawson': {'team': 'Racing Bulls', 'role_context': 'Racing Bulls driver',
                       'aliases': ['Liam', 'Liam Lawson']},
            'Lindblad': {'team': 'Racing Bulls', 'role_context': 'Racing Bulls rookie',
                         'aliases': ['Arvid', 'Arvid Lindblad']},
            'Hulkenberg': {'team': 'Audi', 'role_context': 'Audi experienced driver',
                           'aliases': ['Nico', 'Nico Hulkenberg', 'Hülkenberg']},
            'Bortoleto': {'team': 'Audi', 'role_context': 'Audi rookie driver',
                          'aliases': ['Gabriel', 'Gabriel Bortoleto']},
            'Ocon': {'team': 'Haas', 'role_context': 'Haas lead driver', 'aliases': ['Esteban', 'Esteban Ocon']},
            'Bearman': {'team': 'Haas', 'role_context': 'Haas young driver',
                        'aliases': ['Oliver', 'Oliver Bearman', 'Ollie']},
            'Sainz': {'team': 'Williams', 'role_context': 'Williams lead driver',
                      'aliases': ['Carlos', 'Carlos Sainz']},
            'Albon': {'team': 'Williams', 'role_context': 'Williams experienced driver',
                      'aliases': ['Alex', 'Alexander Albon']},
            'Alonso': {'team': 'Aston Martin', 'role_context': 'Aston Martin veteran champion',
                       'aliases': ['Fernando', 'Fernando Alonso']},
            'Stroll': {'team': 'Aston Martin', 'role_context': 'Aston Martin driver',
                       'aliases': ['Lance', 'Lance Stroll']},
            'Perez': {'team': 'Cadillac', 'role_context': 'Cadillac experienced driver',
                      'aliases': ['Sergio', 'Checo', 'Sergio Perez']},
            'Bottas': {'team': 'Cadillac', 'role_context': 'Cadillac veteran driver',
                       'aliases': ['Valtteri', 'Valtteri Bottas']}
        },
        'teams': {
            'Red Bull': {'aliases': ['Red Bull Racing', 'RBR']},
            'Mercedes': {'aliases': ['Mercedes AMG', 'Mercedes-AMG']},
            'Ferrari': {'aliases': ['Scuderia Ferrari', 'Maranello']},
            'McLaren': {'aliases': ['McLaren F1', 'McLaren Racing']},
            'Alpine': {'aliases': ['Alpine F1', 'Alpine Renault']},
            'Racing Bulls': {'aliases': ['Racing Bulls', 'RB', 'Visa Cash App RB']},
            'Audi': {'aliases': ['Audi F1', 'Audi Sauber', 'Sauber']},
            'Haas': {'aliases': ['Haas F1', 'Haas Ferrari']},
            'Williams': {'aliases': ['Williams Racing', 'Williams F1']},
            'Aston Martin': {'aliases': ['Aston Martin F1', 'Aston Martin Racing']},
            'Cadillac': {'aliases': ['Cadillac F1', 'Cadillac Racing', 'GM Cadillac']}
        }
    }

    context_to_drivers = {
        "Red Bull world champion": "Verstappen", "Red Bull rookie driver": "Hadjar",
        "Mercedes young driver": "Antonelli", "Mercedes lead driver": "Russell",
        "Ferrari lead driver": "Leclerc", "Ferrari seven-time champion": "Hamilton",
        "McLaren world champion": "Norris", "McLaren race winner": "Piastri",
        "Alpine lead driver": "Gasly", "Alpine rookie driver": "Colapinto",
        "Racing Bulls driver": "Lawson", "Racing Bulls rookie": "Lindblad",
        "Audi experienced driver": "Hulkenberg", "Audi rookie driver": "Bortoleto",
        "Haas lead driver": "Ocon", "Haas young driver": "Bearman",
        "Williams lead driver": "Sainz", "Williams experienced driver": "Albon",
        "Aston Martin veteran champion": "Alonso", "Aston Martin driver": "Stroll",
        "Cadillac experienced driver": "Perez", "Cadillac veteran driver": "Bottas"
    }

    def __init__(self, site_link, period):
        self.site_link = site_link
        self.period = period
        self.links = []
        self.finbert_model = pipeline("text-classification", model="ProsusAI/finbert", truncation=True, max_length=512,
                                      device=-1)
        self.driver_scores = {}
        self.driver_news_count = {}

    def summarize_with_sumy(self, text, max_words=150):
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.lsa import LsaSummarizer
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LsaSummarizer()
            sentences_count = max(3, max_words // 15)
            summary = summarizer(parser.document, sentences_count)
            return " ".join([str(s) for s in summary])
        except:
            return None

    def summarize_by_truncation(self, text, max_words=150):
        words = text.split()
        return " ".join(words[:max_words]) if len(words) > max_words else text

    def evaluate_text(self, text):
        results = self.finbert_model(text, candidate_labels=["positive", "negative", "neutral"])
        label = results[0]['label']
        score = results[0]['score']
        if label == "neutral":
            return None
        confidence = score if label == "positive" else -score
        drivers = self.extract_drivers_from_context(text)
        if drivers:
            return {d: confidence / len(drivers) for d in drivers}
        return None

    def merge_scores(self, scores1, scores2):
        if scores1 is None and scores2 is None:
            return {}
        if scores1 is None:
            return scores2
        if scores2 is None:
            return scores1
        merged = {}
        for d in set(scores1.keys()) | set(scores2.keys()):
            v1 = scores1.get(d, 0)
            v2 = scores2.get(d, 0)
            merged[d] = (v1 + v2) / 2
        return merged

    def enhance_text_with_context(self, text):
        enhanced_text = text
        for driver_name, driver_info in self.entity_dict.get('drivers', {}).items():
            pattern = re.compile(r'\b' + re.escape(driver_name) + r'\b', re.IGNORECASE)
            if pattern.search(enhanced_text):
                team = driver_info['team']
                role = driver_info.get('role_context', f"{team} driver")
                text_lower = text.lower()
                if any(w in text_lower for w in ['win', 'victory', 'won', 'champion']):
                    enhanced_text = f"{text} This is positive for {team} brand value. regarding {role}"
                elif any(w in text_lower for w in ['crash', 'failure', 'dnf', 'retired']):
                    enhanced_text = f"{text} This may negatively impact {team} reputation. regarding {role}"
                else:
                    enhanced_text = f"{text} regarding {team} performance. regarding {role}"
                break
            else:
                for alias in driver_info.get("aliases", []):
                    pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
                    if pattern.search(enhanced_text):
                        team = driver_info['team']
                        role = driver_info.get('role_context', f"{team} driver")
                        text_lower = text.lower()
                        if any(w in text_lower for w in ['win', 'victory', 'won']):
                            enhanced_text = f"{text} This is positive for {team} brand value. regarding {role}"
                        elif any(w in text_lower for w in ['crash', 'failure', 'dnf']):
                            enhanced_text = f"{text} This may negatively impact {team} reputation. regarding {role}"
                        else:
                            enhanced_text = f"{text} regarding {team} performance. regarding {role}"
                        break
        return enhanced_text

    def extract_drivers_from_context(self, text):
        drivers = []
        for key, value in self.context_to_drivers.items():
            if key in text and value not in drivers:
                drivers.append(value)
        return drivers

    def get_predictions(self):
        self.links = self.get_links()

        for i, link_info in enumerate(self.links):
            link = link_info['url'] if isinstance(link_info, dict) else link_info

            try:
                full_text = self.get_text_by_link(link)
                if not full_text:
                    continue

                text_sumy = self.summarize_with_sumy(full_text)
                text_trunc = self.summarize_by_truncation(full_text)

                scores_sumy = self.evaluate_text(self.enhance_text_with_context(text_sumy)) if text_sumy else None
                scores_trunc = self.evaluate_text(self.enhance_text_with_context(text_trunc))

                scores = self.merge_scores(scores_sumy, scores_trunc)

                for driver, score in scores.items():
                    self.driver_scores[driver] = self.driver_scores.get(driver, 0) + score
                    self.driver_news_count[driver] = self.driver_news_count.get(driver, 0) + 1


            except Exception as e:
                continue

            time.sleep(0.5)

        final = {d: self.driver_scores[d] / self.driver_news_count[d]
                 for d in self.driver_scores if self.driver_news_count[d] > 0}
        return final

    @abstractmethod
    def get_links(self):
        pass

    @abstractmethod
    def get_text_by_link(self, link):
        pass

    def get_sentiment_dataframe(self):
        if not self.driver_scores:
            self.get_predictions()
        if not self.driver_scores:
            return pd.DataFrame(columns=['driver', 'total_score', 'news_count', 'average_sentiment'])
        data = []
        for d in self.driver_scores:
            avg = self.driver_scores[d] / max(self.driver_news_count.get(d, 1), 1)
            data.append({'driver': d, 'total_score': self.driver_scores[d],
                         'news_count': self.driver_news_count.get(d, 0), 'average_sentiment': avg})
        df = pd.DataFrame(data)
        return df.sort_values('average_sentiment', ascending=False) if not df.empty else df