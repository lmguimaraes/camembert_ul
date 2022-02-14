import os
import sqlite3

class ServiceArticles:    
    def _obtenir_articles(self):
        listeArticles = []
        try:
            sqliteConnection = sqlite3.connect(self.fichierBD)
            cursor = sqliteConnection.cursor()
            print("Base de données créée et connectée avec succès à SQLite")
            sqlite_select_Query = self.queryObtenirArticles
            cursor.execute(sqlite_select_Query)
            results = cursor.fetchall()
            for result in results:
                listeArticles.append(result)           
            cursor.close()
            return listeArticles

        except sqlite3.Error as error:
            print("SQLite erreur:", error)
        finally:
            if sqliteConnection:
                sqliteConnection.close()
                print("La connexion SQLite est fermée")

    def _obtenir_article_par_id(self, idArticle):
        try:
            sqliteConnection = sqlite3.connect(self.fichierBD)
            cursor = sqliteConnection.cursor()
            print("Base de données créée et connectée avec succès à SQLite")
            sqlite_select_Query = self.queryObtenirArticleParId.format(idArticle)
            cursor.execute(sqlite_select_Query)
            result = cursor.fetchall()   
            cursor.close()
            return result

        except sqlite3.Error as error:
            print("SQLite erreur:", error)
        finally:
            if sqliteConnection:
                sqliteConnection.close()
                print("La connexion SQLite est fermée")            
        
    fichierBD = "{0}/news_database.sqlite".format(os.path.dirname(__file__))
    queryObtenirArticles = "SELECT * FROM news_table"
    queryObtenirArticleParId = "SELECT * FROM news_table WHERE post_id = '{}'"



