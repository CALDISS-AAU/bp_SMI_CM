#!/usr/bin/env python
# coding: utf-8

#Loading packages
import sys
import os
from os.path import join
import pysbd
# redirect stdout and stderr. Needed for loading Setfit packages.
if sys.stdout is None or sys.stderr is None:
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w') 
from setfit import SetFitModel
from gooey import Gooey, GooeyParser
from pipeline_functions import *

# Function to get base directory depending on whether the script is bundled
def get_base_dir():
    if getattr(sys, 'frozen', False):  # If the script is bundled with PyInstaller
        base_dir = sys._MEIPASS  # Temporary folder created by PyInstaller.
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Development environment
    return base_dir

# Get the base directory
base_dir = get_base_dir()

# Define the path of the fine-tuned model
model_dir = os.path.join(base_dir, 'rep_speech_model')

# Ensure that the directory exists and print it for debugging
#if not os.path.exists(model_dir):
#    raise FileNotFoundError(f"Model directory not found: {model_dir}")
#print(f"Loading model from: {model_dir}")

# Load the fine-tuned model from the local path and force local files
classifier = SetFitModel.from_pretrained(model_dir, local_files_only=True)

# Full pipeline
def process_text(input_file, output_dir, highlight_label, classifier = classifier, highlight_color = "yellow", segmenter = pysbd.Segmenter(language="da", clean=False)):

    # output file/path from input_file name
    infile_name = os.path.basename(input_file).replace(".docx", "")
    output_name = infile_name + "_processed" + ".html"
    output_file = join(output_dir, output_name)

    # docx to string
    text = read_docx_to_string(input_file)

    # convert to sentences
    sentences = segmenter.segment(text)

    # predict using model/classifier (expects one label per text)
    predictions = classifier.predict(sentences)

    # store labels
    labels = predictions

    # zip sentences with labels
    sentences_with_labels = [{'sentence': sentence, 'label': label} for sentence, label in zip(sentences, labels)]

    # highlight sentences based on label and join together as text
    highlight_text = highlight_sentences(sentences_with_labels, highlight_label)

    # convert to html
    text_html = generate_html(highlight_text)

    # Write the HTML content to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(text_html)

    # convert to docx
    convert_html_to_docx(output_file)


# Gooey GUI program wrapper for script
@Gooey(
    program_name='Reported speech analysis program', #<- Naming the program
    advanced= True, #<- advanced mode that allows customization and specific points
    menu=[{ # defining the menu and contents here
        'name': 'Help',
        'items': [{
            'type': 'MessageDialog',
            'menuTitle': 'Program Documentation',
            'caption': 'Program Documentation',
            'message': 'The program requires an input .docx file and a designated, existing output folder. When these requirements are met, the program initiates the analysis using the fine-tuned SetFit model (rep-speech model), predicts the labels, and highlights reported speech. Finally, the program converts the provided file to a new .docx file in the specified output folder as [filename]_processed.docx.'
        }]
    },
    {
        'name': 'About',
        'items': [{
            'type': 'AboutDialog',
            'menuTitle': 'About',
            'name': 'Reported Speech Analysis Program',
            'description': 'A program for analyzing reported speech using a fine-tuned SetFit model.',
            'version': '1.0.0',
            'copyright': '2024',
            'website': 'https://www.caldiss.aau.dk/'
        }]
    }],
    progress_regex=r"^progress: (?P<current>\d+)/(?P<total>\d+)$", # progress bar when program is running
       progress_expr="current / total * 100",
       timing_options = {
        'show_time_remaining':True,
        'hide_time_remaining_on_complete':True,
    }
)
#main function expected by Gooey
def main():
    parser = GooeyParser(description="Run Few-shot analysis on .docx document and highlight reported speech") #<- defining description of the program
    
    # Defining parsers for final .exe program
    parser.add_argument('input_file', metavar='Input File',
                        help="Select the file to process",
                        widget='FileChooser') # input file box
    parser.add_argument('output_folder', metavar= 'Output directory',
                        help="Select the directory to save the transformed file",
                        widget='DirChooser') # output folder box

    args = parser.parse_args() #parsing arguments
    print(f"Selected file: {args.input_file}")
    print(f"Output Directory: {args.output_folder}")

    # Call the main function with the provided directories
    process_text(args.input_file, args.output_folder, highlight_label='reported speech')
    
if __name__ == "__main__":
    main()