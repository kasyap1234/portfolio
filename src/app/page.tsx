'use client'
import { Button } from '@/components/ui/button'
import Link from 'next/link'
import {Avatar, AvatarGroup, AvatarIcon} from "@nextui-org/avatar";
import {User} from "@nextui-org/user";
import SkillCard from '@/components/SkillCard';
import {Image} from "@nextui-org/image";
import Typewriter from 'typewriter-effect';
export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center space-y-4 text-center">
      <h1 className="text-4xl font-bold">
        I'm{' '}
        <span className="inline-block">
          <Typewriter
            options={{
             strings: ['Kasyap Dharanikota 👋'],
              autoStart: true,
              loop: true,
              delay: 75,
              deleteSpeed: 50,
              pauseFor: 2500,
            }}
          />
        </span>
      </h1>
      <p className="text-xl text-muted-foreground">
        I'm a passionate software developer specializing in backend development . 
        I love building scalable and efficient solutions.
        I have experience in a variety of technologies, including Node.js, Express, and MongoDB, Golang, Docker, Kubernetes. 
        I'm always eager to learn and stay up-to-date with the latest trends in the industry.
      </p>
  <Image
  isBlurred
  isZoomed
  className="rounded-large mb-10 mt-10 "
      alt="NextUI hero Image with delay"
      height={250}
      src="profile.jpg"
      width={250}
    />

      <div className="flex space-x-4 mt-5 mb-10 ">
        <Button asChild>
          <Link href="/projects">View Projects</Link>
        </Button>
        <Button asChild variant="outline">
          <Link href="/about">About Me</Link>
        </Button>
        </div>
<h2 className="text-2xl font-bold">Skills</h2>
<SkillCard /> 
      
    </div>
  )
}

