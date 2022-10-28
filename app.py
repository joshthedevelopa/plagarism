import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/results", methods=['POST'])
def results():
    if request.method == 'POST':
        file = request.files['fileToUpload']
        file.save('media/' + file.filename)

    return render_template('results.html', context=check_plagiarism())


def similarity(doc1, doc2): return cosine_similarity([doc1, doc2])

def vectorize(Text): return TfidfVectorizer().fit_transform(Text).toarray()

def check_plagiarism():
    global s_vectors

    student_files = [doc for doc in os.listdir("media/") if doc.endswith('.txt')]
    student_notes = [open("media/" + _file, encoding='utf-8').read() for _file in student_files]

    vectors = vectorize(student_notes)
    s_vectors = list(zip(student_files, vectors))

    plagiarism_results = {}
    for student_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))

        del new_vectors[current_index]

        for student_b, text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]

            if(sim_score > 0):
                sim_score = round(sim_score, 1)
                student_pair = sorted((os.path.splitext(student_a)[0], os.path.splitext(student_b)[0]))
                res = (student_pair[0]+' similar to '+ student_pair[1])
                plagiarism_results[res] = sim_score

    return plagiarism_results