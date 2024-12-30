import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import Link from 'next/link'
import {Image} from "@nextui-org/image";
interface ProjectCardProps {
  title: string
  description: string
  technologies: string[]
  link: string
  image: string
}

export function ProjectCard({ title, description, technologies, link,image }: ProjectCardProps) {
  return (
    <Card className="overflow-hidden">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <Image  isBlurred src={image} className="w-full mt-5 mb-5" alt={title}
      /> 
      <CardContent>
        <div className="flex flex-wrap gap-2">
          {technologies.map((tech) => (
            <Badge key={tech} >
              {tech}
            </Badge>
          ))}
        </div>
        <Link
          href={link}
          className="mt-4 inline-block text-sm font-medium text-primary hover:underline"
          target="_blank"
          rel="noopener noreferrer"
        >
          View Project
        </Link>
      </CardContent>
    </Card>
  )
}

