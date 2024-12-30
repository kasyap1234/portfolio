import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

const experiences = [
  {
    title: 'Software Engineer',
    company: 'Thoughtgreen Technologies',
    period: 'September 2024 - Present',
    location: 'Hyderabad, India',
    responsibilities: [
      'Building AI applications using Langchain and Llamaindex. Using langchain to build a RAG',
      'Automating processes using AI applications and AI agents to complete tasks intelligently',
    ],
  },
  {
    title: 'Software Engineer/Intern',
    company: 'Comtek Solutions',
    period: 'July 2023 – July 2024',
    location: 'Hyderabad, India',
    responsibilities: [
      'Used Prometheus for infrastructure monitoring and downtime reporting',
      'Architected and deployed Dockerized back-end application on AWS EC2, integrated with MongoDB on AWS',
      'Used App Signal for performance monitoring and analytics',
      'Built crypto chain with proof of ownership, implemented provider dashboard using Golang',
      'Developed NestJS-based administration system improving payment processing efficiency by 20%',
    ],
  },
  {
    title: 'Software Engineer Intern',
    company: 'Jugyah',
    period: '2023',
    location: 'Mumbai, India',
    responsibilities: [
      'Developed website using ReactJS, Material UI, ExpressJS, NodeJS',
      'Created Python-based price recommendation model with 95% accuracy',
      'Implemented AI-based system with NLP and LLM for personalized recommendations',
      'Improved page load speed by 40% and reduced query time by 20% through database sharding',
    ],
  },
]

const projects = [
  {
    title: 'NPM CLI ENV Package',
    description: 'Developed and published NPM package for managing environment variables using AES encryption',
    link: 'link'
  },
  {
    title: 'Bandwidth Monitoring',
    description: 'Web application for network bandwidth monitoring with ExpressJS backend and Material UI frontend',
    link: 'link'
  },
  {
    title: 'Image Dehazing Using Visual Transformers',
    description: 'DehazeFormer model based on Swin Transformer with custom normalization and activation functions',
    link: 'link'
  },
  {
    title: 'Workflow Automation Agent',
    description: 'Contextually-aware automated workflow system using AI agents and Gemini API for intelligent task execution',
    link: 'link'
  },
]

const education = {
  degree: 'Bachelor of Technology in Artificial Intelligence',
  university: 'Mahindra University',
  year: '2024',
}

const contact = {
  name: 'Kasyap Dharanikota',
  phone: '9872256432',
  email: 'kasyap3103@gmail.com',
  linkedin: 'Linkedin',
  github: 'github.com/kasyap1234'
}

const skills = {
  languages: ['Go', 'TypeScript', 'Javascript', 'C++', 'Python'],
  technologies: ['React.js', 'Express.js', 'TensorFlow', 'NestJS', 'MongoDB', 'Svelte', 'Node.js', 'Docker', 'Kubernetes', 'gRPC', 'Helm'],
  others: ['AWS', 'Swagger API', 'Firebase SDK', 'GIT', 'Twilio', 'Linux', 'PostgreSQL', 'LLM', 'CockroachDB', 'MongoDB', 'OpenAI API', 'Langchain', 'LlamaIndex']
}

export default function Resume() {
  return (
    <div className="space-y-6 p-8">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold">{contact.name}</h1>
        <div className="text-sm text-muted-foreground mt-2">
          <p>{contact.phone} | {contact.email}</p>
         <div className="flex justify-center space-x-4">
        <ion-icon name="logo-github"></ion-icon>

        </div>
        </div> 
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Education</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="font-semibold">{education.degree}</p>
          <p className="text-sm text-muted-foreground">{education.university}, {education.year}</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Experience</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {experiences.map((exp) => (
            <div key={exp.title} className="space-y-2">
              <div className="flex justify-between items-start">
                <h3 className="text-lg font-semibold">{exp.title}</h3>
                <span className="text-sm text-muted-foreground">{exp.period}</span>
              </div>
              <p className="text-md font-medium">{exp.company} - {exp.location}</p>
              <ul className="list-disc list-inside text-sm space-y-1">
                {exp.responsibilities.map((resp, index) => (
                  <li key={index}>{resp}</li>
                ))}
              </ul>
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Projects</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {projects.map((project) => (
            <div key={project.title} className="space-y-2">
              <h3 className="text-lg font-semibold">{project.title}</h3>
              <p className="text-sm">{project.description}</p>
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Technical Skills</CardTitle>
        </CardHeader>
        <CardContent>
          {Object.entries(skills).map(([category, items]) => (
            <div key={category} className="mb-4">
              <h3 className="text-md font-semibold capitalize mb-2">{category}:</h3>
              <p className="text-sm">{items.join(', ')}</p>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  )
}
