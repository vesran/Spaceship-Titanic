pipeline {
    agent {
        dockerfile {
            filename 'Dockerfile'
            dir '.'
            label 'jenkins-build'
            args '-v /app:.'
        }
    }

    stages {
        stage('Test') {
            steps {
                pytest -s tests/test_evaluate.py
            }
        }
    }

}