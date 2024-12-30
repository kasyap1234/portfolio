
import ContactForm from './contact-form'
import { Card, CardContent } from "@/components/ui/card"
import { Github, Linkedin, Twitter } from 'lucide-react'
import Link from 'next/link'

export default function ContactPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8">Contact Us</h1>
      <div className="grid md:grid-cols-2 gap-8">
        <div>
          <ContactForm />
        </div>
        <div className="space-y-6">
          <Card>
            <CardContent className="p-6">
              <h2 className="text-2xl font-semibold mb-4">Get in Touch</h2>
              <p className="mb-2">We'd love to hear from you. Feel free to reach out with any questions or inquiries.</p>
              <p className="mb-2"><strong>Email:</strong> contact@devwebsite.com</p>
              <p><strong>Phone:</strong> +1 (123) 456-7890</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6">
              <h2 className="text-2xl font-semibold mb-4">Follow Us</h2>
              <div className="flex space-x-4">
                <Link href="https://github.com" className="text-gray-600 hover:text-gray-900">
                  <Github size={24} />
                  <span className="sr-only">GitHub</span>
                </Link>
                <Link href="https://linkedin.com" className="text-gray-600 hover:text-gray-900">
                  <Linkedin size={24} />
                  <span className="sr-only">LinkedIn</span>
                </Link>
                <Link href="https://twitter.com" className="text-gray-600 hover:text-gray-900">
                  <Twitter size={24} />
                  <span className="sr-only">Twitter</span>
                </Link>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

