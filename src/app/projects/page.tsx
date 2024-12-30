import { ProjectCard } from '@/components/project-card'

const projects = [
  {
    title: 'URL shortner and stats microservice',
    description: 'two microservices one for url shortner and other for stats,caching to prevent repeated calls to the dragonfly db  and get results directly from the cache',  
    technologies: ['golang', 'docker', 'kubernetes','GRPC','dragonflydb','postgresql','redis'],
    link: 'https://github.com/kasyap1234/url-shortner-microservice',
    image: 'urlshortener.webp'
  },
  {
    title: 'Task Management App',
    description: 'A responsive task management application with real-time updates.',
    technologies: ['React', 'Firebase', 'Tailwind CSS'],
    link: 'https://github.com/yourusername/task-management-app',
  },
  {
    title: 'Weather Dashboard',
    description: 'A weather dashboard that displays current conditions and forecasts.',
    technologies: ['Vue.js', 'OpenWeatherMap API', 'Chart.js'],
    link: 'https://github.com/yourusername/weather-dashboard',
  },
]

export default function Projects() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">My Projects</h1>
      <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {projects.map((project) => (
          <ProjectCard key={project.title} {...project} />
        ))}
      </div>
    </div>
  )
}

