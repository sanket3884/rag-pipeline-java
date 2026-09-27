pipeline {
    agent any

    stages {
        stage('Hello') {
            steps {
                echo 'Jenkins pipeline is working!'
            }
        }
    }
}


pipeline{
    agent any

    tools{
        maven 'Maven-3.9'
    }

    stages{
        stage('Build'){
            steps{
                'mvn clean package -DskipTests'
            }
        }

        stage('Docker Build'){
            steps{
                sh 'docker build -t rag-pipeline:latest'
            }
        }
    }
}