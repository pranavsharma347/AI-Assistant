# serializers.py
from rest_framework import serializers

class FileQuestionSerializer(serializers.Serializer):
    file_uploaded = serializers.FileField()#for file upload from browser here by default required =True 
    question = serializers.CharField()#for question by default required =True means file can not blank when submit



#for uploading multiple file here ListFiled+Field take multiple for upload
class MultiFileUploadSerializer(serializers.Serializer):
    file_uploaded = serializers.ListField(
        child=serializers.FileField(),
        allow_empty=False
    )
    question = serializers.CharField(required=True)



    #for multiple urls

class MultiURLQuestionSerializer(serializers.Serializer):
    urls = serializers.ListField(
        child=serializers.URLField(),
        allow_empty=False,
        help_text="List of document or web URLs"
    )
    question = serializers.CharField(required=True)



class AIGeneratorSerializer(serializers.Serializer):
    question=serializers.CharField(required=True)

