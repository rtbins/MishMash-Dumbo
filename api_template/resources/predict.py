from flask_restful import Resource, reqparse
from flask import Flask, request, Response
import pandas as pd
import os


class Predict(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument("data",
                        type=str,
                        required=True,
                        help="image data to parse"
                        )

    def get(self):
        # returns all the available models
        pass
    
    def post(self):
        # runs a model based on the input and returns the predictions
        pass


# ----------------HELPER FUNCIONS------------------------------------------
